"""Host-only lifecycle contracts for the private zoom presenter permit."""
import unittest

from Renderer.native.native_cpp_test import ROOT, run_cpp


STUB = r'''
#include <cassert>
#include <cstdint>
#include <type_traits>
using DWORD = std::uint32_t;
using HRESULT = std::int32_t;
struct Signal { bool fired = false; bool closed = false; };
using HANDLE = Signal*;
constexpr DWORD ERROR_SUCCESS = 0, ERROR_NOT_SUPPORTED = 50, ERROR_INVALID_DATA = 13;
constexpr DWORD WAIT_OBJECT_0 = 0, WAIT_ABANDONED = 128, WAIT_TIMEOUT = 258;
constexpr DWORD WAIT_FAILED = 0xffffffffu, QS_ALLINPUT = 0x4ff, MWMO_INPUTAVAILABLE = 4;
constexpr HRESULT S_OK = 0, S_FALSE = 1, E_FAIL = static_cast<HRESULT>(0x80004005u);
constexpr HRESULT OCCLUDED = 0x087a0001;
DWORD api_error = 0, single_result = WAIT_TIMEOUT, message_result = WAIT_TIMEOUT;
bool force_single = false, force_message = false;
unsigned single_calls = 0, message_calls = 0, close_calls = 0;
DWORD last_timeout = 0;
bool CloseHandle(HANDLE signal) {
    assert(signal && !signal->closed);
    signal->closed = true;
    ++close_calls;
    return true;
}
DWORD GetLastError() { return api_error; }
DWORD WaitForSingleObject(HANDLE signal, DWORD timeout) {
    assert(signal && !signal->closed && timeout == 0);
    ++single_calls;
    if (force_single) return single_result;
    if (!signal->fired) return WAIT_TIMEOUT;
    signal->fired = false;
    return WAIT_OBJECT_0;
}
DWORD MsgWaitForMultipleObjectsEx(DWORD count, const HANDLE* handles,
                                 DWORD timeout, DWORD mask, DWORD flags) {
    assert(count == 1 && handles && *handles && !(*handles)->closed);
    assert(mask == QS_ALLINPUT && flags == MWMO_INPUTAVAILABLE);
    ++message_calls;
    last_timeout = timeout;
    if (force_message) return message_result;
    if (!(*handles)->fired) return WAIT_TIMEOUT;
    (*handles)->fired = false;
    return WAIT_OBJECT_0;
}
'''


class ZoomPresentationPermitTests(unittest.TestCase):
    def check_cpp(self, body):
        header = (ROOT / "Renderer/sandbox/zoom_presentation_permit.h").read_text()
        # Compile the actual policy with WinAPI stubs; removing this include
        # ensures native_cpp_test selects the host compiler, never the VM.
        header = header.replace("#pragma once", "").replace("#include <windows.h>", "")
        program = STUB + header + r'''
using c3x_renderer::sandbox::ZoomPresentationPermit;
using c3x_renderer::sandbox::ZoomPresentationWait;
int main() {
''' + body + "\n}\n"
        self.assertNotIn("#include <windows.h>", program)
        run_cpp(program)

    def test_initial_reset_ownership_and_no_op_retention(self):
        self.check_cpp(r'''
static_assert(!std::is_copy_constructible<ZoomPresentationPermit>::value);
static_assert(!std::is_copy_assignable<ZoomPresentationPermit>::value);
Signal first, second;
{
    ZoomPresentationPermit permit;
    assert(!permit.valid() && !permit.ready());
    assert(permit.wait(17) == ZoomPresentationWait::unsupported);
    assert(permit.last_error() == ERROR_NOT_SUPPORTED);
    assert(single_calls == 0 && message_calls == 0);
    assert(!permit.presented());
    permit.reset(&first);
    assert(permit.valid() && permit.last_error() == ERROR_SUCCESS);
    assert(!permit.ready());
    first.fired = true;
    assert(permit.ready() && !first.fired);
    const unsigned queries = single_calls;
    for (unsigned i = 0; i < 8; ++i) {
        assert(permit.ready());
        assert(permit.wait(0) == ZoomPresentationWait::granted);
    }
    assert(single_calls == queries && message_calls == 0);
    assert(permit.presented());
    assert(!permit.ready());
    permit.reset(&first);
    assert(!first.closed && close_calls == 0);
    permit.reset(&second);
    assert(first.closed && close_calls == 1);
    second.fired = true;
    assert(permit.wait(20) == ZoomPresentationWait::granted);
    permit.reset();
    assert(second.closed && close_calls == 2 && !permit.valid());
}
assert(close_calls == 2);
Signal final_signal;
{ ZoomPresentationPermit permit; permit.reset(&final_signal); }
assert(final_signal.closed && close_calls == 3);
''')

    def test_wait_latches_grant_and_message_wake_can_retry(self):
        self.check_cpp(r'''
Signal signal;
ZoomPresentationPermit permit;
permit.reset(&signal);
assert(permit.wait(37) == ZoomPresentationWait::timeout);
assert(last_timeout == 37 && !permit.ready());
force_message = true;
message_result = WAIT_OBJECT_0 + 1;
assert(permit.wait(500) == ZoomPresentationWait::message);
assert(!permit.ready() && permit.last_error() == ERROR_SUCCESS);
// The caller drains pending messages and retries without a renderer/input lock.
force_message = false;
signal.fired = true;
const unsigned queries = single_calls;
assert(permit.wait(500) == ZoomPresentationWait::granted);
assert(!signal.fired);
assert(permit.ready() && single_calls == queries);
const unsigned waits = message_calls;
assert(permit.wait(500) == ZoomPresentationWait::granted);
assert(message_calls == waits);
assert(permit.presented());
assert(permit.wait(0) == ZoomPresentationWait::timeout);
assert(message_calls == waits + 1);
signal.fired = true;
assert(permit.wait(0xffffffffu) == ZoomPresentationWait::granted);
assert(last_timeout == 0xffffffffu);
''')

    def test_present_consumes_only_exact_success(self):
        self.check_cpp(r'''
Signal signal;
ZoomPresentationPermit permit;
permit.reset(&signal);
assert(!permit.presented(S_OK));
signal.fired = true;
assert(permit.wait(0) == ZoomPresentationWait::granted);
const unsigned queries = single_calls, waits = message_calls;
const HRESULT statuses[] = {E_FAIL, S_FALSE, OCCLUDED};
for (const HRESULT result : statuses) {
    assert(!permit.presented(result));
    assert(permit.ready());
    assert(permit.wait(0) == ZoomPresentationWait::granted);
}
assert(single_calls == queries && message_calls == waits);
assert(permit.presented(S_OK));
assert(!permit.presented(S_OK));
assert(!permit.ready());
signal.fired = true;
assert(permit.ready());
assert(permit.presented());
assert(permit.wait(0) == ZoomPresentationWait::timeout);
''')

    def test_wait_errors_are_explicit_and_reset_recovers(self):
        self.check_cpp(r'''
Signal first, recovered;
ZoomPresentationPermit permit;
permit.reset(&first);
force_message = true;
message_result = WAIT_FAILED;
api_error = 1234;
assert(permit.wait(100) == ZoomPresentationWait::failed);
assert(permit.last_error() == 1234 && !permit.ready());
message_result = WAIT_ABANDONED;
assert(permit.wait(100) == ZoomPresentationWait::failed);
assert(permit.last_error() == ERROR_INVALID_DATA);
force_single = true;
single_result = WAIT_FAILED;
bool caught = false;
try { permit.ready(); } catch (const std::runtime_error&) { caught = true; }
assert(caught && permit.last_error() == 1234);
single_result = WAIT_ABANDONED;
caught = false;
try { permit.ready(); } catch (const std::runtime_error&) { caught = true; }
assert(caught && permit.last_error() == ERROR_INVALID_DATA);
permit.reset(&recovered);
assert(first.closed && permit.last_error() == ERROR_SUCCESS);
force_single = force_message = false;
recovered.fired = true;
assert(permit.wait(100) == ZoomPresentationWait::granted);
assert(permit.presented());
permit.reset();
assert(permit.wait(100) == ZoomPresentationWait::unsupported);
assert(permit.last_error() == ERROR_NOT_SUPPORTED);
''')


if __name__ == "__main__":
    unittest.main()

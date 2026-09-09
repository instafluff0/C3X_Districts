// Offline bytecode inspection only; this tool never loads a game executable.
#include <d3dcompiler.h>
#include <fstream>
#include <iterator>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    std::ifstream input(argv[1],std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(input),{}};
    ID3DBlob* text=nullptr;
    if(bytes.empty() || FAILED(D3DDisassemble(bytes.data(),bytes.size(),0,nullptr,&text)))return 1;
    std::ofstream output(argv[2],std::ios::binary);
    output.write(static_cast<char const*>(text->GetBufferPointer()),text->GetBufferSize());
    text->Release();return output?0:1;
}

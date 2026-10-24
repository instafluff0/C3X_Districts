using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

// Lab-only companion of Renderer/tools/asset_compiler/export_civ6_animation.cs.
// A Civ VI build movie reveals and hides its pieces with per-bone vector tracks
// ("<bone>_visibility"), which the Firaxis managed wrapper does not expose. The
// wrapper keeps the native granny_track_group (packed: Name @0,
// VectorTrackCount @8, VectorTracks @12; each granny_vector_track is Name @0,
// TrackKey @8, Dimension @12, granny_curve2 ValueCurve @16, 32 bytes), so each
// curve is sampled by pointing a borrowed GrannyCurve at it.
// Output: one line per vector track: group, name, dimension, then the frames
// where the sampled value changes ("frame:value[,value...]").
internal static class ExportVisibility
{
    private static unsafe int Main(string[] args)
    {
        if (args.Length != 3)
        {
            Console.Error.WriteLine("usage: export_visibility <CivNexus6.exe> <raw-granny.fgx> <output.txt>");
            return 2;
        }
        string temporary = null;
        try
        {
            string converter = Path.GetFullPath(args[0]);
            temporary = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".fgx");
            File.Copy(Path.GetFullPath(args[1]), temporary);
            object loader;
            object file = Load(converter, temporary, out loader);
            object animation = ((IList)Get(file, "Animations"))[0];
            float duration = Convert.ToSingle(Get(animation, "Duration"));
            float step = Convert.ToSingle(Get(animation, "TimeStep"));
            int frames = (int)Math.Round(duration / step) + 1;
            StringBuilder output = new StringBuilder();
            output.Append("duration=" + duration.ToString("R", CultureInfo.InvariantCulture) + " frames=" + frames + "\n");
            foreach (object group in (IList)Get(animation, "TrackGroups"))
            {
                string groupName = Convert.ToString(Get(group, "Name"));
                IList transforms = (IList)Get(group, "TransformTracks");
                FieldInfo native = group.GetType().GetField("m_pkTrackGroup", BindingFlags.NonPublic | BindingFlags.Instance);
                byte* raw = (byte*)Pointer.Unbox(native.GetValue(group));
                int count = *(int*)(raw + 8);
                byte* array = *(byte**)(raw + 12);
                if (count <= 0 || array == null || transforms.Count == 0)
                {
                    Console.WriteLine("group " + groupName + " vector_tracks=0");
                    continue;
                }
                // Granny's own evaluator (linked into the C++/CLI wrapper, reachable on its <Module>).
                Module module = transforms[0].GetType().Assembly.ManifestModule;
                MethodInfo evaluate = module.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                    .Single(method => method.Name == "GrannyEvaluateCurveAtT");
                Type curvePointer = evaluate.GetParameters()[3].ParameterType;
                float* result = stackalloc float[16];
                float* identity = stackalloc float[16];
                for (int index = 0; index < count; index++)
                {
                    byte* track = array + index * 32;
                    string name = Marshal.PtrToStringAnsi(*(IntPtr*)track);
                    int dimension = *(int*)(track + 12);
                    if (dimension < 1 || dimension > 16)
                        throw new InvalidDataException("Unexpected vector track dimension " + dimension + " for " + name);
                    output.Append(groupName + "\t" + name + "\t" + dimension + "\t");
                    object curvePtr = Pointer.Box(track + 16, curvePointer);
                    string previous = null;
                    for (int frame = 0; frame < frames; frame++)
                    {
                        float time = Math.Min(duration, frame * step);
                        for (int k = 0; k < 16; k++) { result[k] = 0.0f; identity[k] = 0.0f; }
                        evaluate.Invoke(null, new object[] { dimension, false, false, curvePtr, false, duration, time,
                                                             Pointer.Box(result, typeof(float*)), Pointer.Box(identity, typeof(float*)) });
                        StringBuilder value = new StringBuilder();
                        for (int k = 0; k < dimension; k++)
                            value.Append((k > 0 ? "," : "") + result[k].ToString("0.###", CultureInfo.InvariantCulture));
                        string text = value.ToString();
                        if (text != previous)
                            output.Append(frame + ":" + text + " ");
                        previous = text;
                    }
                    output.Append("\n");
                }
                Console.WriteLine("group " + groupName + " vector_tracks=" + count);
            }
            File.WriteAllText(Path.GetFullPath(args[2]), output.ToString());
            GC.KeepAlive(file);
            GC.KeepAlive(loader);
            return 0;
        }
        catch (Exception error)
        {
            while (error is TargetInvocationException && error.InnerException != null)
                error = error.InnerException;
            Console.Error.WriteLine("export_visibility: " + error);
            return 1;
        }
        finally
        {
            if (temporary != null)
            {
                try { File.Delete(temporary); } catch { }
            }
        }
    }

    private static object Get(object target, string name)
    {
        PropertyInfo property = target.GetType().GetProperty(name);
        object value = property == null ? null : property.GetValue(target, null);
        if (value == null)
            throw new InvalidDataException("Missing property " + name + " on " + target.GetType().Name);
        return value;
    }

    private static object Load(string converter, string input, out object loader)
    {
        string directory = Path.GetDirectoryName(converter);
        Assembly utility = Assembly.LoadFrom(Path.Combine(directory, "Firaxis.Utility.dll"));
        Type available = utility.GetType("Firaxis.Utility.Available", true);
        using (var key = Microsoft.Win32.Registry.CurrentUser.CreateSubKey(@"Software\Firaxis\Tools"))
        {
            object previous = key.GetValue("ToolAssetPath");
            if (previous == null)
            {
                string fallback = Path.Combine(Path.GetTempPath(), "C3XAnimationExporter");
                Directory.CreateDirectory(Path.Combine(fallback, "Assets"));
                Directory.CreateDirectory(Path.Combine(fallback, "Resource"));
                key.SetValue("ToolAssetPath", fallback);
            }
            try
            {
                available.GetMethod("Startup", new Type[] { typeof(string), typeof(bool) }).Invoke(
                    null, new object[] { "C3XVisibilityExporter", true });
            }
            finally
            {
                if (previous == null)
                    key.DeleteValue("ToolAssetPath", false);
            }
        }
        Assembly granny = Assembly.LoadFrom(Path.Combine(directory, "Firaxis.Granny.Impl.dll"));
        Type loaderType = granny.GetTypes().Single(type => type.Name == "GrannyFileLoader");
        loader = Activator.CreateInstance(loaderType);
        object file = loaderType.GetMethod("LoadGrannyFile").Invoke(loader, new object[] { input });
        if (file == null)
            throw new InvalidDataException("Firaxis Granny loader returned no file");
        return file;
    }
}

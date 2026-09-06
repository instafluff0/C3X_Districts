using System;
using System.IO;
using System.Linq;
using System.Reflection;

// Offline helper: turn a minimal CN6 skeleton companion into a Granny model
// that can be supplied to IGrannyAnimation.SampleBone. The emitted FGX is an
// intermediate conversion artifact and is never a runtime dependency.
internal static class ImportCn6Model
{
    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Length != 3)
        {
            Console.Error.WriteLine("usage: import_cn6_model <CivNexus6.exe> <input.cn6> <output.fgx>");
            return 2;
        }
        try
        {
            string converter = Path.GetFullPath(args[0]);
            string input = Path.GetFullPath(args[1]);
            string output = Path.GetFullPath(args[2]);
            if (!File.Exists(converter) || !File.Exists(input))
                throw new FileNotFoundException("Missing converter or CN6 input");

            Assembly assembly = Assembly.LoadFrom(converter);
            Type formType = assembly.GetType("NexusBuddy.CivNexusSixApplicationForm", true);
            object form = Activator.CreateInstance(formType);
            formType.GetField("form", BindingFlags.Public | BindingFlags.Static).SetValue(null, form);

            Type operationsType = assembly.GetType("NexusBuddy.FileOps.CN6FileOps", true);
            MethodInfo import = operationsType.GetMethod(
                "importCN6", BindingFlags.Public | BindingFlags.Static);
            if (import == null)
                throw new MissingMethodException("CivNexus6 CN6 import method was not found");
            Type contextType = import.GetParameters()[2].ParameterType;
            object context = Activator.CreateInstance(contextType);
            Exception deferredError = null;
            try
            {
                import.Invoke(null, new object[] { input, output, context, 0 });
            }
            catch (Exception error)
            {
                // Headless WinForms cannot select the mesh row used by CivNexus's
                // optional material-binding pass. The model is saved immediately
                // before that pass, so accept it only after loading and validating
                // the saved skeleton through the Firaxis API below.
                deferredError = Unwrap(error);
            }
            if (!File.Exists(output))
                throw deferredError ?? new IOException("CivNexus6 did not create " + output);
            object modelFile = contextType.GetMethod("LoadGrannyFile").Invoke(
                context, new object[] { output });
            object models = modelFile.GetType().GetProperty("Models").GetValue(modelFile, null);
            int modelCount = (int)models.GetType().GetProperty("Count").GetValue(models, null);
            if (modelCount != 1)
                throw new InvalidDataException("Sampling companion must contain exactly one model");
            object model = models.GetType().GetProperty("Item").GetValue(models, new object[] { 0 });
            object skeleton = model.GetType().GetProperty("Skeleton").GetValue(model, null);
            object bones = skeleton.GetType().GetProperty("Bones").GetValue(skeleton, null);
            int boneCount = (int)bones.GetType().GetProperty("Count").GetValue(bones, null);
            if (boneCount < 1)
                throw new InvalidDataException("Sampling companion contains no bones");
            Console.WriteLine("models=1 bones=" + boneCount);
            if (deferredError != null)
                Console.WriteLine("material_binding=skipped_headless");
            Console.WriteLine(output);
            GC.KeepAlive(form);
            GC.KeepAlive(context);
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine("import_cn6_model: " + Unwrap(error));
            return 1;
        }
    }

    private static Exception Unwrap(Exception error)
    {
        while (error is TargetInvocationException && error.InnerException != null)
            error = error.InnerException;
        return error;
    }
}

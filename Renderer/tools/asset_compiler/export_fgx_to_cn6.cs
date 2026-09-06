using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;

internal static class ExportFgxToCn6
{
    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Length != 2)
        {
            Console.Error.WriteLine("usage: export_fgx_to_cn6 <CivNexus6.exe> <input.fgx>");
            return 2;
        }

        string converter = Path.GetFullPath(args[0]);
        string input = Path.GetFullPath(args[1]);
        if (!File.Exists(converter) || !File.Exists(input))
        {
            Console.Error.WriteLine("export_fgx_to_cn6: missing converter or FGX input");
            return 2;
        }

        try
        {
            Assembly assembly = Assembly.LoadFrom(converter);
            Type formType = assembly.GetType("NexusBuddy.CivNexusSixApplicationForm", true);
            object form = FormatterServices.GetUninitializedObject(formType);
            formType.GetField("form", BindingFlags.Public | BindingFlags.Static).SetValue(null, form);

            string converterDirectory = Path.GetDirectoryName(converter);
            Assembly utilityAssembly = Assembly.LoadFrom(
                Path.Combine(converterDirectory, "Firaxis.Utility.dll"));
            Type availableType = utilityAssembly.GetType("Firaxis.Utility.Available", true);
            availableType.GetMethod("Startup", new Type[] { typeof(string), typeof(bool) }).Invoke(
                null, new object[] { "CivNexus6", true });

            Assembly grannyAssembly = Assembly.LoadFrom(
                Path.Combine(converterDirectory, "Firaxis.Granny.Impl.dll"));
            Type loaderType = grannyAssembly.GetTypes().Single(type => type.Name == "GrannyFileLoader");
            object loader = Activator.CreateInstance(loaderType);
            object granny = loaderType.GetMethod("LoadGrannyFile").Invoke(loader, new object[] { input });
            if (granny == null)
                throw new InvalidDataException("CivNexus6 returned no Granny file");

            int modelCount = GetCollectionCount(granny, "Models");
            int meshCount = GetCollectionCount(granny, "Meshes");
            Console.WriteLine("models=" + modelCount + " meshes=" + meshCount);
            if (modelCount == 0)
                throw new InvalidDataException("FGX contains no exportable Granny models");

            Type operationsType = assembly.GetType("NexusBuddy.FileOps.CN6FileOps", true);
            MethodInfo export = operationsType.GetMethod(
                "exportAllModelsToCN6", BindingFlags.Public | BindingFlags.Static);
            if (export == null)
                throw new MissingMethodException("CivNexus6 CN6 export method was not found");
            export.Invoke(null, new object[] { granny });

            string output = Path.ChangeExtension(input, ".cn6");
            if (!File.Exists(output))
                throw new IOException("CivNexus6 did not create " + output);
            Console.WriteLine(output);
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine("export_fgx_to_cn6: " + error);
            return 1;
        }
    }

    private static int GetCollectionCount(object owner, string propertyName)
    {
        object collection = owner.GetType().GetProperty(propertyName).GetValue(owner, null);
        return (int)collection.GetType().GetProperty("Count").GetValue(collection, null);
    }
}

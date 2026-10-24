using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;

// Lab-only companion of export_visibility.cs. Civ VI wonder fires and lights
// attach to bones (FX_bone_Fire_Perm_01, ...) that the package's own bone
// arrays leave out; the Granny file itself carries them. This lists every
// skeleton in the file through the Firaxis managed wrapper: per bone its name,
// parent index and every transform-like property the wrapper exposes, so the
// Lab can place the effects. With "--probe" it lists the wrapper's property
// names instead (the file, its skeletons and their first bone).
internal static class ExportBones
{
    private static int Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("usage: export_bones <CivNexus6.exe> <raw-granny.fgx> <output.txt> [--probe]");
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
            StringBuilder output = new StringBuilder();
            bool probe = args.Length > 3 && args[3] == "--probe";
            if (probe)
                Describe(output, "file", file);
            foreach (string listName in new[] { "Skeletons", "Models" })
            {
                PropertyInfo property = file.GetType().GetProperty(listName);
                if (property == null)
                    continue;
                IList items = property.GetValue(file, null) as IList;
                if (items == null)
                    continue;
                for (int index = 0; index < items.Count; index++)
                {
                    object item = items[index];
                    object skeleton = listName == "Models" ? TryGet(item, "Skeleton") : item;
                    if (skeleton == null)
                        continue;
                    IList bones = TryGet(skeleton, "Bones") as IList;
                    string name = Convert.ToString(TryGet(skeleton, "Name"));
                    output.Append("skeleton\t" + listName + "\t" + index + "\t" + name + "\t" + (bones == null ? 0 : bones.Count) + "\n");
                    if (probe)
                    {
                        Describe(output, "skeleton", skeleton);
                        if (bones != null && bones.Count > 0)
                            Describe(output, "bone", bones[0]);
                        continue;
                    }
                    if (bones == null)
                        continue;
                    for (int b = 0; b < bones.Count; b++)
                    {
                        object bone = bones[b];
                        output.Append(b + "\t" + Convert.ToString(TryGet(bone, "Name")) + "\t" +
                                      Convert.ToString(TryGet(bone, "ParentIndex")));
                        foreach (PropertyInfo field in bone.GetType().GetProperties())
                        {
                            if (field.Name == "Name" || field.Name == "ParentIndex" || field.GetIndexParameters().Length > 0)
                                continue;
                            object value = SafeGet(field, bone);
                            output.Append("\t" + field.Name + "=" + Flatten(value));
                        }
                        output.Append("\n");
                    }
                }
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
            Console.Error.WriteLine("export_bones: " + error);
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

    private static void Describe(StringBuilder output, string label, object target)
    {
        foreach (PropertyInfo property in target.GetType().GetProperties())
        {
            if (property.GetIndexParameters().Length > 0)
                continue;
            object value = SafeGet(property, target);
            string text = value is IList ? "list[" + ((IList)value).Count + "]" : Flatten(value);
            if (text.Length > 160)
                text = text.Substring(0, 160);
            output.Append("probe\t" + label + "\t" + property.Name + "\t" + property.PropertyType.Name + "\t" + text + "\n");
        }
    }

    private static object SafeGet(PropertyInfo property, object target)
    {
        try { return property.GetValue(target, null); }
        catch { return null; }
    }

    private static object TryGet(object target, string name)
    {
        PropertyInfo property = target.GetType().GetProperty(name);
        return property == null ? null : SafeGet(property, target);
    }

    // Numbers as invariant text; arrays and float-bearing structs as their fields.
    private static string Flatten(object value)
    {
        if (value == null)
            return "";
        if (value is float || value is double)
            return Convert.ToDouble(value).ToString("R", CultureInfo.InvariantCulture);
        if (value is string || value.GetType().IsPrimitive || value is Enum)
            return Convert.ToString(value, CultureInfo.InvariantCulture);
        IEnumerable sequence = value as IEnumerable;
        if (sequence != null)
            return "[" + string.Join(",", sequence.Cast<object>().Take(32).Select(Flatten)) + "]";
        PropertyInfo[] properties = value.GetType().GetProperties().Where(p => p.GetIndexParameters().Length == 0).ToArray();
        if (properties.Length > 0 && properties.Length <= 24)
            return "{" + string.Join(",", properties.Select(p => p.Name + ":" + Flatten(SafeGet(p, value)))) + "}";
        return value.ToString();
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
                    null, new object[] { "C3XBoneExporter", true });
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

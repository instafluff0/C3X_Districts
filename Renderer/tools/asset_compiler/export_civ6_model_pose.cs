using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;

// Offline-only model-aware Civ VI animation baker. The output contains named,
// frame-major world matrices and has no Granny/Firaxis runtime dependency.
internal static class ExportCiv6ModelPose
{
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("C3XPOSE\0");

    private static int Main(string[] args)
    {
        if (args.Length != 5)
        {
            Console.Error.WriteLine("usage: export_civ6_model_pose <CivNexus6.exe> <animation> <model.fgx> <output.c3pose> <translation-scale>");
            return 2;
        }
        string temporaryAnimation = null;
        try
        {
            string converter = Path.GetFullPath(args[0]);
            string animationInput = Path.GetFullPath(args[1]);
            string modelInput = Path.GetFullPath(args[2]);
            string output = Path.GetFullPath(args[3]);
            float translationScale = Single.Parse(args[4], CultureInfo.InvariantCulture);
            if (!(translationScale > 0.0f) || Single.IsNaN(translationScale) || Single.IsInfinity(translationScale))
                throw new InvalidDataException("Translation scale must be positive and finite");
            if (!File.Exists(converter) || !File.Exists(animationInput) || !File.Exists(modelInput))
                throw new FileNotFoundException("Missing converter, animation, or model input");

            temporaryAnimation = WriteRawAnimation(animationInput);
            string directory = Path.GetDirectoryName(converter);
            Assembly utilityAssembly = Assembly.LoadFrom(Path.Combine(directory, "Firaxis.Utility.dll"));
            utilityAssembly.GetType("Firaxis.Utility.Available", true)
                .GetMethod("Startup", new Type[] { typeof(string), typeof(bool) })
                .Invoke(null, new object[] { "C3XModelPoseExporter", true });
            Assembly grannyAssembly = Assembly.LoadFrom(Path.Combine(directory, "Firaxis.Granny.Impl.dll"));
            Type loaderType = grannyAssembly.GetTypes().Single(type => type.Name == "GrannyFileLoader");
            object loader = Activator.CreateInstance(loaderType);
            MethodInfo load = loaderType.GetMethod("LoadGrannyFile");
            object animationFile = load.Invoke(loader, new object[] { temporaryAnimation });
            object modelFile = load.Invoke(loader, new object[] { modelInput });
            IList animations = AsList(GetProperty(animationFile, "Animations"));
            IList models = AsList(GetProperty(modelFile, "Models"));
            if (animations.Count != 1 || models.Count != 1)
                throw new InvalidDataException("Expected exactly one animation and one model");
            object animation = animations[0];
            object model = models[0];
            float duration = Convert.ToSingle(GetProperty(animation, "Duration"));
            float timeStep = Convert.ToSingle(GetProperty(animation, "TimeStep"));
            if (!(duration > 0.0f) || !(timeStep > 0.0f))
                throw new InvalidDataException("Animation duration or timestep is invalid");
            uint frameCount = checked((uint)Math.Round(duration / timeStep) + 1u);
            float sampleRate = 1.0f / timeStep;
            IList bones = AsList(GetProperty(GetProperty(model, "Skeleton"), "Bones"));
            if (bones.Count < 1)
                throw new InvalidDataException("Sampling model contains no bones");
            MethodInfo sampleBone = animation.GetType().GetMethod("SampleBone");
            if (sampleBone == null)
                throw new MissingMethodException("IGrannyAnimation.SampleBone is unavailable");

            List<string> names = new List<string>();
            foreach (object bone in bones)
            {
                string name = Convert.ToString(GetProperty(bone, "Name"));
                if (String.IsNullOrEmpty(name) || name.IndexOf('\0') >= 0 || names.Contains(name))
                    throw new InvalidDataException("Sampling model contains an invalid or duplicate bone name");
                names.Add(name);
            }
            WritePose(output, animation, model, sampleBone, names, duration, timeStep,
                sampleRate, frameCount, translationScale);
            Console.WriteLine("duration=" + duration.ToString("R", CultureInfo.InvariantCulture));
            Console.WriteLine("sample_rate=" + sampleRate.ToString("R", CultureInfo.InvariantCulture));
            Console.WriteLine("frames=" + frameCount + " bones=" + names.Count);
            Console.WriteLine("output_bytes=" + new FileInfo(output).Length);
            Console.WriteLine(output);
            GC.KeepAlive(animationFile);
            GC.KeepAlive(modelFile);
            GC.KeepAlive(loader);
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine("export_civ6_model_pose: " + Unwrap(error));
            return 1;
        }
        finally
        {
            if (temporaryAnimation != null)
            {
                try { File.Delete(temporaryAnimation); }
                catch { }
            }
        }
    }

    private static string WriteRawAnimation(string input)
    {
        byte[] source = File.ReadAllBytes(input);
        string temporary = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".fgx");
        if (source.Length >= 17 && Encoding.ASCII.GetString(source, 0, 6) == "CIVBIG")
        {
            uint payloadBytes = BitConverter.ToUInt32(source, 8);
            if (payloadBytes == 0 || payloadBytes > source.Length - 16)
                throw new InvalidDataException("CIVBIG payload length is invalid");
            byte[] payload = new byte[source.Length - 16];
            Buffer.BlockCopy(source, 16, payload, 0, payload.Length);
            File.WriteAllBytes(temporary, payload);
        }
        else if (source.Length >= 16 && BitConverter.ToString(source, 0, 16).Replace("-", "").ToLowerInvariant() == "e59b495e6f631f141e13eba990beedc4")
        {
            File.WriteAllBytes(temporary, source);
        }
        else
        {
            throw new InvalidDataException("Unsupported animation payload");
        }
        return temporary;
    }

    private static void WritePose(string output, object animation, object model,
        MethodInfo sampleBone, List<string> names, float duration, float timeStep,
        float sampleRate, uint frameCount, float translationScale)
    {
        List<byte[]> nameBytes = names.Select(Encoding.UTF8.GetBytes).ToList();
        uint recordsOffset = 36;
        uint stringsOffset = checked(recordsOffset + (uint)names.Count * 8u);
        uint samplesOffset = stringsOffset;
        foreach (byte[] value in nameBytes)
            samplesOffset = checked(samplesOffset + (uint)value.Length);
        samplesOffset = (samplesOffset + 3u) & ~3u;
        Directory.CreateDirectory(Path.GetDirectoryName(output));
        using (BinaryWriter writer = new BinaryWriter(File.Create(output), Encoding.UTF8))
        {
            writer.Write(Magic);
            writer.Write((uint)1);
            writer.Write(duration);
            writer.Write(sampleRate);
            writer.Write(frameCount);
            writer.Write((uint)names.Count);
            writer.Write(recordsOffset);
            writer.Write(samplesOffset);
            uint nameOffset = stringsOffset;
            foreach (byte[] value in nameBytes)
            {
                writer.Write(nameOffset);
                writer.Write((uint)value.Length);
                nameOffset += (uint)value.Length;
            }
            foreach (byte[] value in nameBytes)
                writer.Write(value);
            while (writer.BaseStream.Position < samplesOffset)
                writer.Write((byte)0);
            for (uint frame = 0; frame < frameCount; frame++)
            {
                float time = Math.Min(duration, frame * timeStep);
                foreach (string name in names)
                {
                    float[] matrix = null;
                    Exception lastError = null;
                    for (int attempt = 0; attempt < 3 && matrix == null; attempt++)
                    {
                        try { matrix = (float[])sampleBone.Invoke(animation, new object[] { model, name, time }); }
                        catch (Exception error) { lastError = Unwrap(error); }
                    }
                    if (matrix == null)
                        throw new InvalidDataException("SampleBone failed for " + name + " at frame " + frame, lastError);
                    if (matrix.Length != 16)
                        throw new InvalidDataException("SampleBone returned a non-4x4 matrix");
                    for (int component = 0; component < 16; component++)
                    {
                        float value = matrix[component];
                        if (component == 12 || component == 13 || component == 14)
                            value *= translationScale;
                        if (Single.IsNaN(value) || Single.IsInfinity(value))
                            throw new InvalidDataException("SampleBone returned a non-finite matrix");
                        if (Math.Abs(value) < 1.0e-12f)
                            value = 0.0f;
                        writer.Write(value);
                    }
                }
            }
        }
    }

    private static object GetProperty(object value, string name)
    {
        return value.GetType().GetProperty(name).GetValue(value, null);
    }

    private static IList AsList(object value)
    {
        IList list = value as IList;
        if (list == null)
            throw new InvalidDataException("Expected Granny list");
        return list;
    }

    private static Exception Unwrap(Exception error)
    {
        while (error is TargetInvocationException && error.InnerException != null)
            error = error.InnerException;
        return error;
    }
}

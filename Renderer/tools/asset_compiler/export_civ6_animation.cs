using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Text;

// Offline-only Civ VI cooked-animation importer.  The output intentionally has
// no Granny/Firaxis dependency; see normalized_animation.py for the consumer.
internal static class ExportCiv6Animation
{
    private const int HeaderBytes = 56;
    private const int GroupRecordBytes = 16;
    private const int TrackRecordBytes = 32;
    private const uint FormatVersion = 1;
    private const uint Identity = 0;
    private const uint Constant = 1;
    private const uint Sampled = 2;

    private sealed class Channel
    {
        public uint Mode;
        public int Dimension;
        public List<float> Values = new List<float>();
        public uint FileOffset;
    }

    private sealed class Track
    {
        public string Name;
        public uint Flags;
        public uint NameOffset;
        public uint NameBytes;
        public Channel Position;
        public Channel Orientation;
        public Channel ScaleShear;
    }

    private sealed class Group
    {
        public string Name;
        public uint NameOffset;
        public uint NameBytes;
        public uint FirstTrack;
        public uint TrackCount;
    }

    [HandleProcessCorruptedStateExceptions]
    private static int Main(string[] args)
    {
        if (args.Length != 3 && args.Length != 4)
        {
            Console.Error.WriteLine("usage: export_civ6_animation <CivNexus6.exe> <input> <output.c3anim> [translation-scale]");
            return 2;
        }

        string temporaryFgx = null;
        try
        {
            string converter = Path.GetFullPath(args[0]);
            string input = Path.GetFullPath(args[1]);
            string output = Path.GetFullPath(args[2]);
            float translationScale = args.Length == 4
                ? Single.Parse(args[3], System.Globalization.CultureInfo.InvariantCulture)
                : 1.0f;
            if (!(translationScale > 0.0f) || Single.IsNaN(translationScale) || Single.IsInfinity(translationScale))
                throw new InvalidDataException("Translation scale must be positive and finite");
            if (!File.Exists(converter) || !File.Exists(input))
                throw new FileNotFoundException("Missing converter or animation input");

            byte[] source = File.ReadAllBytes(input);
            temporaryFgx = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".fgx");
            byte[] grannyBytes;
            if (source.Length >= 17 && Encoding.ASCII.GetString(source, 0, 6) == "CIVBIG")
            {
                uint declaredPayloadBytes = BitConverter.ToUInt32(source, 8);
                if (declaredPayloadBytes == 0 || declaredPayloadBytes > source.Length - 16)
                    throw new InvalidDataException("CIVBIG payload length is invalid");
                grannyBytes = new byte[source.Length - 16];
                Buffer.BlockCopy(source, 16, grannyBytes, 0, grannyBytes.Length);
            }
            else if (source.Length >= 16 && BitConverter.ToString(source, 0, 16).Replace("-", "").ToLowerInvariant() == "e59b495e6f631f141e13eba990beedc4")
            {
                grannyBytes = source;
            }
            else
            {
                throw new InvalidDataException("Input is neither a Civ VI CIVBIG animation nor a supported raw Granny payload");
            }
            File.WriteAllBytes(temporaryFgx, grannyBytes);

            object grannyLoader;
            object file = LoadGrannyFile(converter, temporaryFgx, out grannyLoader);
            IList animations = AsList(GetProperty(file, "Animations"), "Animations");
            if (animations.Count != 1)
                throw new InvalidDataException("Expected exactly one animation, found " + animations.Count);

            object animation = animations[0];
            float duration = Convert.ToSingle(GetProperty(animation, "Duration"));
            float timeStep = Convert.ToSingle(GetProperty(animation, "TimeStep"));
            if (!(duration > 0.0f) || !(timeStep > 0.0f))
                throw new InvalidDataException("Animation duration or timestep is invalid");
            uint frameCount = checked((uint)Math.Round(duration / timeStep) + 1u);
            float sampleRate = 1.0f / timeStep;
            if (frameCount < 2 || Math.Abs(((frameCount - 1) * timeStep) - duration) > timeStep * 0.01f)
                throw new InvalidDataException("Duration is not an integral number of source timesteps");

            IList sourceGroups = AsList(GetProperty(animation, "TrackGroups"), "TrackGroups");
            List<Group> groups = new List<Group>();
            List<Track> tracks = new List<Track>();
            for (int groupIndex = 0; groupIndex < sourceGroups.Count; groupIndex++)
            {
                object sourceGroup = sourceGroups[groupIndex];
                IList sourceTracks = AsList(GetProperty(sourceGroup, "TransformTracks"), "TransformTracks");
                Group group = new Group();
                group.Name = Convert.ToString(GetProperty(sourceGroup, "Name"));
                if (String.IsNullOrEmpty(group.Name) || group.Name.IndexOf('\0') >= 0)
                    throw new InvalidDataException("Animation contains an empty or invalid group name");
                group.FirstTrack = checked((uint)tracks.Count);
                group.TrackCount = checked((uint)sourceTracks.Count);
                groups.Add(group);
                Console.WriteLine("source_group[" + groupIndex + "]=" + group.Name + " tracks=" + sourceTracks.Count);
                HashSet<string> names = new HashSet<string>(StringComparer.Ordinal);
                foreach (object sourceTrack in sourceTracks)
                {
                    string name = Convert.ToString(GetProperty(sourceTrack, "Name"));
                    if (String.IsNullOrEmpty(name) || name.IndexOf('\0') >= 0)
                        throw new InvalidDataException("Animation contains an empty or invalid track name");
                    if (!names.Add(name))
                        throw new InvalidDataException("Duplicate transform track name within group " + groupIndex + ": " + name);
                    Track track = new Track();
                    track.Name = name;
                    uint sourceFlags = Convert.ToUInt32(GetProperty(sourceTrack, "Flags"));
                    if (sourceFlags != 0)
                        throw new InvalidDataException("Unsupported non-zero Granny track flags on " + name);
                    track.Flags = 0;
                    track.Position = ReadChannel(GetProperty(sourceTrack, "PositionCurve"), 3, duration, timeStep, frameCount, false, translationScale, name + ".position");
                    track.Orientation = ReadChannel(GetProperty(sourceTrack, "OrientationCurve"), 4, duration, timeStep, frameCount, true, 1.0f, name + ".orientation");
                    track.ScaleShear = ReadChannel(GetProperty(sourceTrack, "ScaleShearCurve"), 9, duration, timeStep, frameCount, false, 1.0f, name + ".scale_shear");
                    tracks.Add(track);
                }
            }
            if (tracks.Count == 0)
                throw new InvalidDataException("Animation contains no transform tracks");

            WriteClip(output, duration, sampleRate, frameCount, groups, tracks);
            Console.WriteLine("clip=" + Convert.ToString(GetProperty(animation, "Name")));
            Console.WriteLine("duration=" + duration.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            Console.WriteLine("sample_rate=" + sampleRate.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            Console.WriteLine("frames=" + frameCount);
            Console.WriteLine("groups=" + groups.Count);
            Console.WriteLine("tracks=" + tracks.Count);
            Console.WriteLine("output_bytes=" + new FileInfo(output).Length);
            Console.WriteLine(output);
            GC.KeepAlive(file);
            GC.KeepAlive(grannyLoader);
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine("export_civ6_animation: " + Unwrap(error));
            return 1;
        }
        finally
        {
            if (temporaryFgx != null)
            {
                try { File.Delete(temporaryFgx); }
                catch { }
            }
        }
    }

    private static object LoadGrannyFile(string converter, string input, out object loader)
    {
        string directory = Path.GetDirectoryName(converter);
        Assembly utilityAssembly = Assembly.LoadFrom(Path.Combine(directory, "Firaxis.Utility.dll"));
        Type availableType = utilityAssembly.GetType("Firaxis.Utility.Available", true);
        availableType.GetMethod("Startup", new Type[] { typeof(string), typeof(bool) }).Invoke(
            null, new object[] { "C3XAnimationExporter", true });

        Assembly grannyAssembly = Assembly.LoadFrom(Path.Combine(directory, "Firaxis.Granny.Impl.dll"));
        Type loaderType = grannyAssembly.GetTypes().Single(type => type.Name == "GrannyFileLoader");
        loader = Activator.CreateInstance(loaderType);
        object file = loaderType.GetMethod("LoadGrannyFile").Invoke(loader, new object[] { input });
        if (file == null)
            throw new InvalidDataException("Firaxis Granny loader returned no file");
        return file;
    }

    [HandleProcessCorruptedStateExceptions]
    private static Channel ReadChannel(object curve, int dimension, float duration, float timeStep, uint frameCount, bool normalize, float valueScale, string label)
    {
        if (Convert.ToInt32(GetProperty(curve, "Dimension")) != dimension)
            throw new InvalidDataException("Unexpected curve dimension");
        Channel channel = new Channel();
        channel.Dimension = dimension;
        if (Convert.ToBoolean(GetProperty(curve, "IsIdentity")))
        {
            channel.Mode = Identity;
            return channel;
        }
        channel.Mode = Convert.ToBoolean(GetProperty(curve, "IsConstant")) ? Constant : Sampled;
        uint samples = channel.Mode == Constant ? 1u : frameCount;
        for (uint frame = 0; frame < samples; frame++)
        {
            float time = channel.Mode == Constant ? 0.0f : Math.Min(duration, frame * timeStep);
            float[] values;
            try
            {
                values = SampleCurveWithRecovery(curve, time, duration, timeStep, normalize, dimension, label, frame);
            }
            catch (InvalidDataException)
            {
                // Some shipped mounted clips use compressed channels that the
                // bundled Granny evaluator cannot decode reliably. A single
                // unrecoverable or implausible sample invalidates that whole
                // channel: mark it absent so normalized skin evaluation uses
                // the corresponding skeleton-rest position, orientation, or
                // scale/shear while preserving every other decoded channel.
                channel.Mode = Identity;
                channel.Values.Clear();
                Console.WriteLine("recovered_rest_channel=" + label + " frame=" + frame);
                return channel;
            }
            foreach (float sourceValue in values)
            {
                float value = sourceValue * valueScale;
                if (Single.IsNaN(value) || Single.IsInfinity(value))
                    throw new InvalidDataException("Curve sampler returned a non-finite value");
                if (Math.Abs(value) < 1.0e-12f)
                    value = 0.0f;
                channel.Values.Add(value);
            }
        }
        return channel;
    }

    [HandleProcessCorruptedStateExceptions]
    private static float[] SampleCurveWithRecovery(object curve, float time, float duration, float timeStep, bool normalize, int dimension, string label, uint frame)
    {
        Exception exactError = null;
        for (int attempt = 0; attempt < 3; attempt++)
        {
            try { return SampleCurve(curve, time, duration, normalize, dimension); }
            catch (Exception error) { exactError = Unwrap(error); }
        }

        // A few shipped resource clips contain a compressed curve knot that
        // Granny's evaluator cannot read at the exact nominal frame time.  Sample
        // immediately around that knot and interpolate across the sub-frame gap.
        // At a clip boundary only one neighboring sample exists, so use that
        // sample instead of asking Granny to evaluate the known-bad endpoint a
        // second time.
        float epsilon = Math.Max(0.000001f, Math.Min(0.0001f, timeStep * 0.001f));
        float beforeTime = Math.Max(0.0f, time - epsilon);
        float afterTime = Math.Min(duration, time + epsilon);
        float[] before = null;
        float[] after = null;
        if (beforeTime < time)
        {
            try { before = SampleCurve(curve, beforeTime, duration, normalize, dimension); }
            catch (Exception) { }
        }
        if (afterTime > time)
        {
            try { after = SampleCurve(curve, afterTime, duration, normalize, dimension); }
            catch (Exception) { }
        }
        try
        {
            if (before == null && after == null)
                throw new InvalidDataException("No valid neighboring curve sample");
            if (before == null || after == null)
            {
                float[] neighbor = before ?? after;
                Console.WriteLine("recovered_neighbor_sample=" + label + " frame=" + frame);
                return neighbor;
            }
            if (normalize && dimension == 4)
            {
                float dot = 0.0f;
                for (int index = 0; index < dimension; index++)
                    dot += before[index] * after[index];
                if (dot < 0.0f)
                    for (int index = 0; index < dimension; index++)
                        after[index] = -after[index];
            }
            float[] recovered = new float[dimension];
            float lengthSquared = 0.0f;
            for (int index = 0; index < dimension; index++)
            {
                recovered[index] = (before[index] + after[index]) * 0.5f;
                lengthSquared += recovered[index] * recovered[index];
            }
            if (normalize && dimension == 4)
            {
                if (!(lengthSquared > 0.0f))
                    throw new InvalidDataException("Recovered quaternion has zero length");
                float inverseLength = 1.0f / (float)Math.Sqrt(lengthSquared);
                for (int index = 0; index < dimension; index++)
                    recovered[index] *= inverseLength;
            }
            Console.WriteLine("recovered_curve_sample=" + label + " frame=" + frame);
            return recovered;
        }
        catch (Exception recoveryError)
        {
            throw new InvalidDataException(
                "Curve sampling failed for " + label + " at frame " + frame + " (t=" + time.ToString("R", System.Globalization.CultureInfo.InvariantCulture) + ")",
                new AggregateException(exactError, Unwrap(recoveryError)));
        }
    }

    private static float[] SampleCurve(object curve, float time, float duration, bool normalize, int dimension)
    {
        MethodInfo sample = curve.GetType().GetMethod("Sample");
        float[] values = (float[])sample.Invoke(curve, new object[] { time, duration, normalize, false, false });
        if (values == null || values.Length != dimension)
            throw new InvalidDataException("Curve sampler returned an unexpected value count");
        if (values.Any(value => Single.IsNaN(value) || Single.IsInfinity(value)))
            throw new InvalidDataException("Curve sampler returned a non-finite value");
        if (values.Any(value => Math.Abs(value) > 1000000.0f))
            throw new InvalidDataException("Curve sampler returned an implausibly large value");
        float lengthSquared = 0.0f;
        foreach (float value in values)
            lengthSquared += value * value;
        if ((dimension == 4 || dimension == 9) && !(lengthSquared > 1.0e-12f))
            throw new InvalidDataException("Curve sampler returned a degenerate orientation or scale/shear value");
        return values;
    }

    private static void WriteClip(string output, float duration, float sampleRate, uint frameCount, List<Group> groups, List<Track> tracks)
    {
        MemoryStream strings = new MemoryStream();
        foreach (Group group in groups)
        {
            byte[] name = Encoding.UTF8.GetBytes(group.Name);
            group.NameOffset = checked((uint)(HeaderBytes + strings.Length));
            group.NameBytes = checked((uint)name.Length);
            strings.Write(name, 0, name.Length);
        }
        foreach (Track track in tracks)
        {
            byte[] name = Encoding.UTF8.GetBytes(track.Name);
            track.NameOffset = checked((uint)(HeaderBytes + strings.Length));
            track.NameBytes = checked((uint)name.Length);
            strings.Write(name, 0, name.Length);
        }
        uint groupTableOffset = Align4(checked((uint)(HeaderBytes + strings.Length)));
        uint trackTableOffset = checked(groupTableOffset + (uint)groups.Count * GroupRecordBytes);
        uint dataOffset = checked(trackTableOffset + (uint)tracks.Count * TrackRecordBytes);
        MemoryStream data = new MemoryStream();
        foreach (Track track in tracks)
        {
            AssignChannelOffset(track.Position, dataOffset, data);
            AssignChannelOffset(track.Orientation, dataOffset, data);
            AssignChannelOffset(track.ScaleShear, dataOffset, data);
        }

        string parent = Path.GetDirectoryName(output);
        if (!String.IsNullOrEmpty(parent))
            Directory.CreateDirectory(parent);
        using (FileStream stream = new FileStream(output, FileMode.Create, FileAccess.Write, FileShare.None))
        using (BinaryWriter writer = new BinaryWriter(stream, Encoding.UTF8))
        {
            writer.Write(Encoding.ASCII.GetBytes("C3XANIM\0"));
            writer.Write(FormatVersion);
            writer.Write(0u);
            writer.Write(duration);
            writer.Write(sampleRate);
            writer.Write(frameCount);
            writer.Write(checked((uint)groups.Count));
            writer.Write(checked((uint)tracks.Count));
            writer.Write(checked((uint)strings.Length));
            writer.Write(groupTableOffset);
            writer.Write(trackTableOffset);
            writer.Write(dataOffset);
            writer.Write(checked((uint)data.Length));
            writer.Write(strings.ToArray());
            while (writer.BaseStream.Position < groupTableOffset)
                writer.Write((byte)0);
            foreach (Group group in groups)
            {
                writer.Write(group.NameOffset);
                writer.Write(group.NameBytes);
                writer.Write(group.FirstTrack);
                writer.Write(group.TrackCount);
            }
            foreach (Track track in tracks)
            {
                writer.Write(track.NameOffset);
                writer.Write(track.NameBytes);
                writer.Write(track.Flags);
                writer.Write(track.Position.Mode | (track.Orientation.Mode << 2) | (track.ScaleShear.Mode << 4));
                writer.Write(track.Position.FileOffset);
                writer.Write(track.Orientation.FileOffset);
                writer.Write(track.ScaleShear.FileOffset);
                writer.Write(0u);
            }
            writer.Write(data.ToArray());
        }
    }

    private static void AssignChannelOffset(Channel channel, uint dataOffset, MemoryStream data)
    {
        if (channel.Mode == Identity)
        {
            channel.FileOffset = 0;
            return;
        }
        channel.FileOffset = checked(dataOffset + (uint)data.Length);
        using (BinaryWriter writer = new BinaryWriter(data, Encoding.UTF8, true))
        {
            foreach (float value in channel.Values)
                writer.Write(value);
        }
    }

    private static uint Align4(uint value)
    {
        return checked((value + 3u) & ~3u);
    }

    private static object GetProperty(object owner, string name)
    {
        PropertyInfo property = owner.GetType().GetProperty(name);
        if (property == null)
            throw new MissingMemberException(owner.GetType().FullName, name);
        return property.GetValue(owner, null);
    }

    private static IList AsList(object value, string name)
    {
        IList list = value as IList;
        if (list == null)
            throw new InvalidDataException(name + " is not an IList");
        return list;
    }

    private static Exception Unwrap(Exception error)
    {
        while (error is TargetInvocationException && error.InnerException != null)
            error = error.InnerException;
        return error;
    }
}

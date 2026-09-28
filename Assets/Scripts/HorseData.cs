using System;
using System.Collections.Generic;
using UnityEngine;

namespace QHop
{
    [Serializable]
    public sealed class HorseConfig
    {
        public int schemaVersion = 1;
        public int seed = 20260927;
        public float bulk = 1f, frontLength = 1f, backLength = 1f, bodyLength = 1f;
        public float frontStrength = 1f, backStrength = 1f, massKg = 450f, hoofSmoothness = .3f;
        public bool independentOverrides;

        public HorseConfig Copy() => JsonUtility.FromJson<HorseConfig>(JsonUtility.ToJson(this));
        public void Clamp()
        {
            bulk = Mathf.Clamp(bulk, .85f, 1.15f);
            frontLength = Mathf.Clamp(frontLength, .85f, 1.15f);
            backLength = Mathf.Clamp(backLength, .85f, 1.15f);
            bodyLength = Mathf.Clamp(bodyLength, .85f, 1.15f);
            frontStrength = Mathf.Clamp(frontStrength, .55f, 1.45f);
            backStrength = Mathf.Clamp(backStrength, .55f, 1.45f);
            massKg = Mathf.Clamp(massKg, 250f, 750f);
            hoofSmoothness = Mathf.Clamp01(hoofSmoothness);
        }
    }

    public static class HorseGenetics
    {
        static float Range(System.Random random) => (float)random.NextDouble() * 2f - 1f;
        public static HorseConfig Generate(int seed)
        {
            var rng = new System.Random(seed);
            float frame = Range(rng), fat = Range(rng);
            float frontMuscle = Range(rng), backMuscle = Range(rng);
            var result = new HorseConfig { seed = seed };
            result.bulk = Mathf.Clamp(1f + .05f * frame + .10f * fat, .85f, 1.15f);
            result.bodyLength = Mathf.Clamp(1f + .10f * frame + .05f * Range(rng), .85f, 1.15f);
            result.frontLength = Mathf.Clamp(1f + .08f * frame + .07f * Range(rng), .85f, 1.15f);
            result.backLength = Mathf.Clamp(1f + .08f * frame + .07f * Range(rng), .85f, 1.15f);
            if (Mathf.Abs(result.frontLength - result.backLength) > .2f)
                result.backLength = result.frontLength + Mathf.Sign(result.backLength - result.frontLength) * .2f;
            result.frontStrength = Mathf.Clamp(1f + .2f * frame + .25f * frontMuscle, .55f, 1.45f);
            result.backStrength = Mathf.Clamp(1f + .2f * frame + .25f * backMuscle, .55f, 1.45f);
            result.massKg = Mathf.Clamp(450f * result.bulk * result.bulk * result.bodyLength *
                (.8f + .1f * (result.frontLength + result.backLength)) * (1f + .05f * Range(rng)), 250f, 750f);
            result.hoofSmoothness = (float)rng.NextDouble();
            return result;
        }
    }

    public sealed class HorseAction
    {
        public const int AngleCount = 16, StrengthCount = 12, Size = 28;
        public readonly float[] values = new float[Size];
        public bool IsValid()
        {
            foreach (float value in values)
                if (float.IsNaN(value) || float.IsInfinity(value) || value < -1f || value > 1f) return false;
            return true;
        }
    }

    public sealed class HorseObservation
    {
        public const int Version = 1;
        public const int Size = 254;
        public readonly List<float> values = new List<float>(256);
        public void Add(Vector3 vector) { values.Add(vector.x); values.Add(vector.y); values.Add(vector.z); }
        public void Add(float value) => values.Add(value);
    }

    public interface IHorseController
    {
        HorseAction Decide(HorseObservation observation);
        string DisplayName { get; }
    }

    public sealed class StandingPoseController : IHorseController
    {
        public string DisplayName => "站立演示控制器";
        public HorseAction Decide(HorseObservation observation)
        {
            var action = new HorseAction();
            for (int i = HorseAction.AngleCount; i < HorseAction.Size; i++) action.values[i] = .7f;
            return action;
        }
    }

    public enum HorseCondition { Standing, Unstable, Fallen, Recovering }
    public enum HorseEventType { Collision, Slip, Fall, Recovery, Boundary, Finish, TestImpulse }
    public struct HorseEvent
    {
        public HorseEventType type;
        public int horseId;
        public Vector3 point;
        public float intensity;
        public float time;
    }
}

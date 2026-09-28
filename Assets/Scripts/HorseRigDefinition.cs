using System;
using UnityEngine;

namespace QHop
{
    [CreateAssetMenu(menuName = "QHOP/Horse Rig Definition")]
    public sealed class HorseRigDefinition : ScriptableObject
    {
        public float referenceShoulderHeight = 1.5f;
        public float rootDrive = 40000f;
        public float jointSpring = 60000f;
        public float jointDamper = 3000f;
        public float passiveSpring = 5000f;
        public float passiveDamper = 350f;
        public float swingLimit = 60f;
        public float sideLimit = 20f;
        public float hingeLimit = 75f;
        public float hoofLimit = 30f;
        public string[] frontLeft = { "ForeThigh.L", "ForeThigh.L.001", "ForeCalf.L", "ForeFoot.L", "ForeToe.L" };
        public string[] frontRight = { "ForeThigh.R", "ForeThigh.R.001", "ForeCalf.R", "ForeFoot.R", "ForeToe.R" };
        public string[] backLeft = { "BackThigh.L", "BackLowerThigh.L", "BackCalf.L", "BackFoot.L", "BackToe.L" };
        public string[] backRight = { "BackThigh.R", "BackLowerThigh.R", "BackCalf.R", "BackFoot.R", "BackToe.R" };
        public string[][] Legs => new[] { frontLeft, frontRight, backLeft, backRight };
    }
}

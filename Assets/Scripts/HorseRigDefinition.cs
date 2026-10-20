using System;
using UnityEngine;

namespace QHop
{
    // Canonical angles: forward swing at the root, flexion at other leg joints.
    // Unity's angular-X limit coordinate is the inverse of the measured body angle.
    public sealed class HorseJointDescriptor
    {
        public ConfigurableJoint Joint {get;private set;}
        public Quaternion Rest {get;private set;}
        public Vector3 AxisInParent {get;private set;}
        public Vector3 SideInParent {get;private set;}
        public float BodySign,Min,Max,SoftMin,SoftMax,Side,Twist;
        public float Target {get;private set;}
        public void Bind(ConfigurableJoint joint)
        {
            Joint=joint;
            Rest=Quaternion.Inverse(joint.connectedBody.rotation)*joint.transform.rotation;
            AxisInParent=Quaternion.Inverse(joint.connectedBody.rotation)*Vector3.right;
            SideInParent=Quaternion.Inverse(joint.connectedBody.rotation)*Vector3.forward;
            float a=-BodySign*Min,b=-BodySign*Max;
            joint.lowAngularXLimit=new SoftJointLimit{limit=Mathf.Min(a,b)};
            joint.highAngularXLimit=new SoftJointLimit{limit=Mathf.Max(a,b)};
            // Secondary axis is up (twist); Z is forward (lateral swing).
            joint.angularYLimit=new SoftJointLimit{limit=Twist};
            joint.angularZLimit=new SoftJointLimit{limit=Side};
        }
        public float ActionAngle(float input) => input*(input>=0f?Max:-Min);
        public void Drive(float canonical,float side=0f)
        {
            Target=canonical;
            Quaternion desired=Quaternion.AngleAxis(side*Side,SideInParent)*
                Quaternion.AngleAxis(BodySign*canonical,AxisInParent)*Rest;
            Vector3 axis=Joint.axis.normalized;
            Vector3 secondary=Vector3.ProjectOnPlane(Joint.secondaryAxis,axis).normalized;
            Quaternion space=Quaternion.LookRotation(Vector3.Cross(axis,secondary),secondary);
            Joint.targetRotation=Quaternion.Inverse(space)*Quaternion.Inverse(desired)*Rest*space;
        }
        public float Measured
        {
            get
            {
                Quaternion current=Quaternion.Inverse(Joint.connectedBody.rotation)*Joint.transform.rotation;
                Quaternion delta=current*Quaternion.Inverse(Rest);
                float projection=Vector3.Dot(new Vector3(delta.x,delta.y,delta.z),AxisInParent);
                return BodySign*Mathf.DeltaAngle(0f,2f*Mathf.Atan2(projection,delta.w)*Mathf.Rad2Deg);
            }
        }
        public float SoftError => Mathf.Clamp01(Measured<SoftMin?
            (SoftMin-Measured)/Mathf.Max(.01f,SoftMin-Min):
            (Measured-SoftMax)/Mathf.Max(.01f,Max-SoftMax));
    }

    [CreateAssetMenu(menuName = "QHOP/Horse Rig Definition")]
    public sealed class HorseRigDefinition : ScriptableObject
    {
        public const string Contract="rig-r14:19rb;18j;28a;mass-preserved;canonical-r13-limits;spring=240000;damper=3000;cap=40000;passive=5000,350;initial-power=.85;solver=64,16;spawn=fourhoof-plane;torso-com-forward=.2*bodyLength";
        public float referenceShoulderHeight=1.5f;
        public float torsoCenterOfMassForward=.2f;
        public float rootDrive=40000f,jointSpring=240000f,jointDamper=3000f;
        public float passiveSpring=5000f,passiveDamper=350f;
        // Retained serialized legacy fields are not used by r13 leg descriptors.
        public float swingLimit=60f,sideLimit=20f,hingeLimit=75f,hoofLimit=30f;
        public float frontBackSwing=45f,frontForwardSwing=55f;
        public float backBackSwing=55f,backForwardSwing=45f;
        public float frontMiddleFlex=75f,frontLowerFlex=100f,backMiddleFlex=60f,backLowerFlex=65f;
        public float frontMiddleSoft=60f,frontLowerSoft=85f,backMiddleSoft=45f,backLowerSoft=50f;
        public float reverseFlexLimit=8f,reverseSoftLimit=3f;
        public float rootSideLimit=12f,rootTwistLimit=8f;
        public float hingeSideLimit=6f,hingeTwistLimit=10f,hoofSideLimit=8f;
        public string[] frontLeft={"ForeThigh.L","ForeThigh.L.001","ForeCalf.L","ForeFoot.L","ForeToe.L"};
        public string[] frontRight={"ForeThigh.R","ForeThigh.R.001","ForeCalf.R","ForeFoot.R","ForeToe.R"};
        public string[] backLeft={"BackThigh.L","BackLowerThigh.L","BackCalf.L","BackFoot.L","BackToe.L"};
        public string[] backRight={"BackThigh.R","BackLowerThigh.R","BackCalf.R","BackFoot.R","BackToe.R"};
        public string[][] Legs => new[]{frontLeft,frontRight,backLeft,backRight};
        public HorseJointDescriptor Describe(int leg,int part)
        {
            bool back=leg>=2;
            if(part==0)return new HorseJointDescriptor{BodySign=-1f,
                Min=-(back?backBackSwing:frontBackSwing),Max=back?backForwardSwing:frontForwardSwing,
                SoftMin=-(back?45f:35f),SoftMax=back?35f:45f,Side=rootSideLimit,Twist=rootTwistLimit};
            if(part==3)return new HorseJointDescriptor{BodySign=1f,Min=-hoofLimit,Max=hoofLimit,
                SoftMin=-25f,SoftMax=25f,Side=hoofSideLimit,Twist=hoofSideLimit};
            return new HorseJointDescriptor{BodySign=back && part==2?-1f:1f,
                Min=-reverseFlexLimit,Max=part==1?(back?backMiddleFlex:frontMiddleFlex):(back?backLowerFlex:frontLowerFlex),
                SoftMin=-reverseSoftLimit,SoftMax=part==1?(back?backMiddleSoft:frontMiddleSoft):(back?backLowerSoft:frontLowerSoft),
                Side=hingeSideLimit,Twist=hingeTwistLimit};
        }
    }
}

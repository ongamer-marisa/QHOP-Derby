using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace QHop
{
    public sealed class HorseActor : MonoBehaviour
    {
        sealed class FollowBone
        {
            public Transform bone;
            public Rigidbody body;
            public Vector3 positionOffset;
            public Quaternion rotationOffset;
        }

        sealed class MotorJoint
        {
            public ConfigurableJoint joint;
            public HorseJointDescriptor descriptor;
            public int angleIndex,strengthIndex;
            public bool root,back;
        }

        public int Id { get; private set; }
        public HorseConfig Config { get; private set; }
        public HorseCondition Condition { get; private set; }
        public Rigidbody Torso { get; private set; }
        public string ControllerName => controller.DisplayName;
        public event Action<HorseEvent> EventRaised;
        public readonly List<Rigidbody> bodies = new List<Rigidbody>();
        public readonly List<ConfigurableJoint> allJoints = new List<ConfigurableJoint>();

        readonly Dictionary<string, Transform> bones = new Dictionary<string, Transform>();
        readonly Dictionary<string, Rigidbody> directBodies = new Dictionary<string, Rigidbody>();
        readonly List<FollowBone> follow = new List<FollowBone>();
        public readonly List<HorseJointDescriptor> LegJoints=new List<HorseJointDescriptor>();
        readonly List<MotorJoint> motors = new List<MotorJoint>();
        readonly List<Collider> ownColliders = new List<Collider>();
        readonly Rigidbody[] hooves = new Rigidbody[4];
        readonly float[] legLengths = new float[4];
        readonly RaycastHit[] groundHits = new RaycastHit[16];
        readonly List<Vector3> restOffsets = new List<Vector3>();
        readonly List<Quaternion> restRotations = new List<Quaternion>();
        readonly bool[] hoofContact = new bool[4];
        readonly HashSet<Collider> torsoGroundContacts = new HashSet<Collider>();
        readonly HashSet<Collider>[] hoofGroundContacts = {
            new HashSet<Collider>(), new HashSet<Collider>(),
            new HashSet<Collider>(), new HashSet<Collider>() };
        readonly Collider[] recoveryHits=new Collider[128];
        HorseRigDefinition rig;
        IHorseController controller = new StandingPoseController();
        HorseAction lastAction = new HorseAction();
        Transform visual;
        Mesh uniqueMesh;
        Material uniqueBodyMaterial;
        Material uniqueOutlineMaterial;
        PhysicsMaterial uniqueHoofMaterial;
        float conditionTimer;
        float lastCollisionEvent = -10f;
        float nextSlipEvent;
        bool torsoOnGround;
        float groundContactTimer;
        float restTorsoHeight;
        float restTorsoClearance;
        bool assistRecovery = true;
        bool trainingMode;
        bool retired;
        Vector3 fallPosition;
        int physicsTick;
        public readonly HorseGaitTracker Gait=new HorseGaitTracker();
        public readonly HorseLidar Lidar=new HorseLidar();
        public int GaitTask {get;set;}=-1;
        public bool ExternalActionMode { get; set; }
        public float FallWaitRemaining => Condition == HorseCondition.Fallen ? Mathf.Max(0f, 3f-conditionTimer) : 0f;
        public bool IsPolicyActive => Condition == HorseCondition.Standing || Condition == HorseCondition.Unstable;
        public IReadOnlyList<float> LastActionValues => lastAction.values;
        public bool[] HoofContact => hoofContact;
        public Rigidbody[] Hooves => hooves;
        public bool TorsoGrounded => torsoOnGround;
        public float ConditionTimer => conditionTimer;
        public float RestTorsoHeight => restTorsoHeight;
        public float RestTorsoClearance => restTorsoClearance;
        public float LegLength(int leg) => legLengths[leg];
        public float JointFlexion(int leg,int part)
        {
            return LegJoints[leg*4+part].Measured;
        }
        public float HoofClearance(int leg)
        {
            var hoof=hooves[leg];
            Vector3 position=hoof.position;
            int count=gameObject.scene.GetPhysicsScene().Raycast(position+Vector3.up*2f,
                Vector3.down,groundHits,4f,Physics.DefaultRaycastLayers,
                QueryTriggerInteraction.Ignore);
            float ground=-.02f;
            for(int i=0;i<count;i++)
                if(groundHits[i].collider.GetComponentInParent<HorseGroundSurface>()!=null)
                    ground=Mathf.Max(ground,groundHits[i].point.y);
            return Mathf.Max(0f,hoof.GetComponent<Collider>().bounds.min.y-ground);
        }

        public void Initialize(int id, HorseConfig config, GameObject horsePrefab, HorseRigDefinition definition,
            Material bodyMaterial, bool allowRecovery, bool training=false)
        {
            Gait.Reset();
            Id = id;
            Config = config.Copy(); Config.Clamp();
            rig = definition;
            assistRecovery = allowRecovery;
            trainingMode=training;
            Condition = HorseCondition.Standing;
            visual = Instantiate(horsePrefab, transform).transform;
            visual.name = "VisibleHorse";
            visual.localPosition = Vector3.zero;
            visual.localRotation = Quaternion.identity;
            visual.localScale = Vector3.one;
            var renderer = visual.GetComponentInChildren<SkinnedMeshRenderer>();
            if (renderer == null) throw new InvalidOperationException("Horse FBX has no skinned mesh.");
            foreach (var bone in renderer.bones)
                if (bone!=null && !bones.ContainsKey(bone.name)) bones[bone.name] = bone;
            if (!bones.ContainsKey("ForeThigh.L") || !bones.ContainsKey("Pelvis"))
                throw new InvalidOperationException("Horse FBX is missing required deform bones.");
            renderer.sharedMesh = Instantiate(renderer.sharedMesh);
            renderer.sharedMesh.name = "HorseMesh_" + id;
            uniqueMesh=renderer.sharedMesh;
            ApplyMuscleMorph(renderer);
            int morph=renderer.sharedMesh.GetBlendShapeIndex(Config.bulk>=1f?"BulkFat":"BulkThin");
            if(morph>=0)renderer.SetBlendShapeWeight(morph,Mathf.Abs(Config.bulk-1f)/.15f*100f);
            var materials = renderer.sharedMaterials;
            materials[0] = Instantiate(bodyMaterial);
            uniqueBodyMaterial=materials[0];
            materials[0].color = new[] { new Color(.64f,.33f,.19f), new Color(.84f,.63f,.35f),
                new Color(.42f,.46f,.54f), new Color(.67f,.43f,.40f) }[id % 4];
            renderer.materials = materials;
            renderer.updateWhenOffscreen = true;
            if(!trainingMode)CreateOutline(renderer,morph);

            foreach (string[] leg in rig.Legs)
            {
                var hip=bones[leg[0]];
                hip.localPosition=new Vector3(hip.localPosition.x*Config.bulk*1.5f,
                    hip.localPosition.y,hip.localPosition.z);
                for (int i = 1; i < leg.Length; i++)
                    bones[leg[i]].localPosition = new Vector3(bones[leg[i]].localPosition.x,
                        bones[leg[i]].localPosition.y * (leg[0].StartsWith("Fore") ? Config.frontLength : Config.backLength),
                        bones[leg[i]].localPosition.z);
            }
            float baseHeight = bones["ForeThigh.L"].position.y - renderer.bounds.min.y;
            float scale = rig.referenceShoulderHeight / Mathf.Max(.01f, baseHeight);
            visual.localScale = new Vector3(scale, scale, scale * Config.bodyLength);
            if ((bones["ForeThigh.L"].position.z - bones["BackThigh.L"].position.z) < 0f)
                visual.localRotation = Quaternion.Euler(0f, 180f, 0f);
            visual.position += Vector3.up * (.025f - renderer.bounds.min.y);
            Vector3 center = (bones["Chest"].position + bones["Pelvis"].position) * .5f;
            visual.position += Vector3.forward * (transform.position.z - center.z);

            CreateBodies(renderer);
            AlignHoofSupportPlane();
            float lowestHoof=float.MaxValue;
            foreach(var hoof in hooves)
                lowestHoof=Mathf.Min(lowestHoof,hoof.GetComponent<Collider>().bounds.min.y);
            // Both flat tracks have a top surface at y=-0.02. Place the
            // lowest hoof collider face exactly on that surface.
            float spawnShift=-.02f-lowestHoof;
            foreach(var body in bodies)body.position+=Vector3.up*spawnShift;
            visual.position+=Vector3.up*spawnShift;
            Physics.SyncTransforms();
            restTorsoHeight = Torso.position.y;
            restTorsoClearance = restTorsoHeight - (-.02f);
            foreach (var body in bodies)
            {
                restOffsets.Add(body.position-Torso.position);
                restRotations.Add(body.rotation);
            }
            if(!trainingMode)CreateBoneFollowers();
            else renderer.enabled=false;
            IgnoreSelfOverlap();
            // Establish all targets and caps before the very first physics tick.
            for(int i=HorseAction.AngleCount;i<HorseAction.Size;i++)lastAction.values[i]=.7f;
            ApplyAction(lastAction);
            Lidar.Reset(this);
        }

        void CreateOutline(SkinnedMeshRenderer source,int activeMorph)
        {
            Shader shader=Shader.Find("QHOP/Outline");
            if(shader==null)return;
            uniqueOutlineMaterial=new Material(shader);
            var go=new GameObject("Soft Ink Outline");
            go.transform.SetParent(source.transform,false);
            var target=go.AddComponent<SkinnedMeshRenderer>();
            target.sharedMesh=source.sharedMesh;
            target.bones=source.bones;
            target.rootBone=source.rootBone;
            target.sharedMaterial=uniqueOutlineMaterial;
            target.updateWhenOffscreen=true;
            target.shadowCastingMode=UnityEngine.Rendering.ShadowCastingMode.Off;
            if(activeMorph>=0)target.SetBlendShapeWeight(activeMorph,source.GetBlendShapeWeight(activeMorph));
        }

        void ApplyMuscleMorph(SkinnedMeshRenderer renderer)
        {
            Mesh mesh=renderer.sharedMesh;
            Vector3[] vertices=mesh.vertices;
            BoneWeight[] weights=mesh.boneWeights;
            Transform[] skinBones=renderer.bones;
            if(weights.Length!=vertices.Length || skinBones.Length==0)return;
            float[] boneX=new float[skinBones.Length];
            int[] boneClass=new int[skinBones.Length];
            for(int i=0;i<skinBones.Length;i++)
            {
                boneX[i]=renderer.transform.InverseTransformPoint(skinBones[i].position).x;
                boneClass[i]=skinBones[i].name.StartsWith("Fore")?1:skinBones[i].name.StartsWith("Back")?2:0;
            }
            for(int i=0;i<vertices.Length;i++)
            {
                BoneWeight w=weights[i];
                int[] indices={w.boneIndex0,w.boneIndex1,w.boneIndex2,w.boneIndex3};
                float[] influences={w.weight0,w.weight1,w.weight2,w.weight3};
                float center=0f, strength=0f, total=0f;
                for(int k=0;k<4;k++)
                {
                    int index=indices[k];
                    if(index<0 || index>=skinBones.Length || boneClass[index]==0)continue;
                    center+=boneX[index]*influences[k];
                    strength+=(boneClass[index]==1?Config.frontStrength:Config.backStrength)*influences[k];
                    total+=influences[k];
                }
                if(total<=.01f)continue;
                center/=total;strength/=total;
                vertices[i].x+=(vertices[i].x-center)*(strength-1f)*.12f*total;
            }
            mesh.vertices=vertices;mesh.RecalculateBounds();
        }

        Rigidbody NewBody(string name, Vector3 center, Quaternion rotation, float massFraction,
            float radius, float length, bool box = false)
        {
            var go = new GameObject(name);
            go.transform.position = center;
            go.transform.rotation = rotation;
            go.transform.SetParent(transform, true);
            var rb = go.AddComponent<Rigidbody>();
            rb.mass = Config.massKg * massFraction;
            rb.linearDamping=.05f;
            rb.angularDamping=box?1.5f:.4f;
            rb.interpolation = RigidbodyInterpolation.Interpolate;
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            rb.solverIterations = 64;
            rb.solverVelocityIterations = 16;
            rb.maxAngularVelocity = 15f;
            if (box)
            {
                var collider = go.AddComponent<BoxCollider>();
                collider.size = new Vector3(radius * 2f, .65f * Config.bulk, length);
                ownColliders.Add(collider);
            }
            else
            {
                var collider = go.AddComponent<CapsuleCollider>();
                collider.direction = 1;
                collider.radius = radius;
                collider.height = Mathf.Max(length, radius * 2.05f);
                ownColliders.Add(collider);
            }
            go.AddComponent<HorseContactRelay>().Owner = this;
            bodies.Add(rb);
            return rb;
        }

        void AlignHoofSupportPlane()
        {
            // Different front/rear lengths need a different whole-body ground pose.
            // Rotate the assembly rigidly: joint rest rotations and limits stay unchanged.
            var center=Torso.position;var positions=new Vector3[bodies.Count];var rotations=new Quaternion[bodies.Count];
            for(int i=0;i<bodies.Count;i++){positions[i]=bodies[i].position;rotations[i]=bodies[i].rotation;}
            float lo=-30f,hi=30f,angle=0f;var visualPosition=visual.position;var visualRotation=visual.rotation;
            for(int iteration=0;iteration<18;iteration++)
            {
                angle=(lo+hi)*.5f;var rotation=Quaternion.AngleAxis(angle,Vector3.right);
                for(int i=0;i<bodies.Count;i++)
                {bodies[i].position=center+rotation*(positions[i]-center);bodies[i].rotation=rotation*rotations[i];}
                Physics.SyncTransforms();
                float front=(hooves[0].GetComponent<Collider>().bounds.min.y+hooves[1].GetComponent<Collider>().bounds.min.y)*.5f;
                float back=(hooves[2].GetComponent<Collider>().bounds.min.y+hooves[3].GetComponent<Collider>().bounds.min.y)*.5f;
                if(front>back)lo=angle;else hi=angle;
            }
            var final=Quaternion.AngleAxis(angle,Vector3.right);
            visual.position=center+final*(visualPosition-center);visual.rotation=final*visualRotation;
        }

        void CreateBodies(SkinnedMeshRenderer renderer)
        {
            Vector3 hip = bones["Pelvis"].position, chest = bones["Chest"].position;
            Vector3 bodyCenter = (hip + chest) * .5f;
            float bodyLength = Mathf.Abs(chest.z - hip.z) + .8f;
            Torso = NewBody("Torso", bodyCenter, Quaternion.identity, .60f, .34f * Config.bulk, bodyLength, true);
            var torsoCollider = Torso.GetComponent<BoxCollider>();
            torsoCollider.center = new Vector3(0f, .08f, 0f);
            Torso.centerOfMass=new Vector3(0f,.08f,rig.torsoCenterOfMassForward*Config.bodyLength);
            Torso.ResetInertiaTensor();
            directBodies["Base"] = Torso;
            directBodies["Pelvis"] = Torso;
            directBodies["Chest"] = Torso;

            var neck = CreateLink("Neck", "NeckBase", "UpperNeck", .08f, .16f * Config.bulk);
            var head = CreateLink("Head", "UpperNeck", "Head", .04f, .18f * Config.bulk);
            Connect(neck, Torso, bones["NeckBase"].position, 35f, false);
            Connect(head, neck, bones["UpperNeck"].position, 45f, false);
            directBodies["NeckBase"] = neck;
            directBodies["UpperNeck"] = head;
            directBodies["Head"] = head;

            for (int legId = 0; legId < 4; legId++)
            {
                string[] chain = rig.Legs[legId];
                bool back = legId >= 2;
                Rigidbody previous = Torso;
                float[] portions = { .4f, .25f, .2f, .15f };
                for (int part = 0; part < 4; part++)
                {
                    Vector3 start = bones[chain[part]].position;
                    Vector3 end = bones[chain[part + 1]].position;
                    Vector3 delta = end - start;
                    var rb = NewBody(chain[part], (start + end) * .5f,
                        Quaternion.FromToRotation(Vector3.up, delta.normalized),
                        .07f * portions[part], part == 3 ? .095f : .075f * Config.bulk,
                        delta.magnitude + (part == 3 ? .08f : 0f));
                    directBodies[chain[part]] = rb;
                    bool active = part < 3;
                    var descriptor=rig.Describe(legId,part);
                    float bodyA=descriptor.BodySign*descriptor.Min,bodyB=descriptor.BodySign*descriptor.Max;
                    var joint=Connect(rb,previous,start,-Mathf.Min(bodyA,bodyB),Mathf.Max(bodyA,bodyB),descriptor.Side,active);
                    descriptor.Bind(joint);LegJoints.Add(descriptor);
                    if(active)motors.Add(new MotorJoint{joint=joint,descriptor=descriptor,
                        angleIndex=legId*4+(part==0?0:part==1?2:3),strengthIndex=legId*3+part,
                        root=part==0,back=back});
                    previous = rb;
                    if (part == 3)
                    {
                        hooves[legId] = rb;
                        if(uniqueHoofMaterial==null)
                            uniqueHoofMaterial = new PhysicsMaterial("Hoof_" + Id)
                            {
                                staticFriction = Mathf.Lerp(.9f, .15f, Config.hoofSmoothness),
                                dynamicFriction = Mathf.Lerp(.7f, .08f, Config.hoofSmoothness),
                                frictionCombine = PhysicsMaterialCombine.Multiply,
                                bounciness = 0f
                            };
                        rb.GetComponent<Collider>().sharedMaterial = uniqueHoofMaterial;
                    }
                }
                directBodies[chain[4]] = previous;
                legLengths[legId]=Vector3.Distance(bones[chain[0]].position,
                    hooves[legId].position);
            }
        }

        Rigidbody CreateLink(string name, string startBone, string endBone, float mass, float radius)
        {
            Vector3 a = bones[startBone].position, b = bones[endBone].position;
            Vector3 delta = b - a;
            return NewBody(name, (a+b)*.5f, Quaternion.FromToRotation(Vector3.up, delta.normalized),
                mass, radius, delta.magnitude);
        }

        ConfigurableJoint Connect(Rigidbody body, Rigidbody parent, Vector3 anchor,
            float limit, bool active)
        { return Connect(body,parent,anchor,limit,limit,limit,active); }

        ConfigurableJoint Connect(Rigidbody body, Rigidbody parent, Vector3 anchor,
            float negativeLimit, float positiveLimit, float sideLimit, bool active)
        {
            var joint = body.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = parent;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = body.transform.InverseTransformPoint(anchor);
            joint.connectedAnchor = parent.transform.InverseTransformPoint(anchor);
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
            joint.axis=body.transform.InverseTransformDirection(Vector3.right);
            joint.secondaryAxis=body.transform.InverseTransformDirection(Vector3.up);
            joint.lowAngularXLimit = new SoftJointLimit { limit = -positiveLimit };
            joint.highAngularXLimit = new SoftJointLimit { limit = negativeLimit };
            joint.angularYLimit = new SoftJointLimit { limit = sideLimit };
            joint.angularZLimit = new SoftJointLimit { limit = sideLimit };
            joint.rotationDriveMode = RotationDriveMode.Slerp;
            joint.slerpDrive = new JointDrive
            {
                positionSpring = active ? rig.jointSpring : rig.passiveSpring,
                positionDamper = active ? rig.jointDamper : rig.passiveDamper,
                maximumForce = active ? rig.rootDrive : rig.passiveSpring
            };
            joint.enableCollision = false;
            joint.projectionMode = JointProjectionMode.PositionAndRotation;
            allJoints.Add(joint);
            return joint;
        }

        void CreateBoneFollowers()
        {
            foreach (var pair in bones)
            {
                Transform node = pair.Value;
                Transform ancestor = node;
                Rigidbody body = Torso;
                while (ancestor != null && ancestor != visual.parent)
                {
                    if (directBodies.TryGetValue(ancestor.name, out body)) break;
                    ancestor = ancestor.parent;
                }
                if (ancestor == null || ancestor == visual.parent) body = Torso;
                follow.Add(new FollowBone { bone = node, body = body,
                    positionOffset = body.transform.InverseTransformPoint(node.position),
                    rotationOffset = Quaternion.Inverse(body.rotation) * node.rotation });
            }
            follow.Sort((a,b) => Depth(a.bone).CompareTo(Depth(b.bone)));
        }

        static int Depth(Transform t) { int n=0; while(t.parent != null) { n++; t=t.parent; } return n; }

        void IgnoreSelfOverlap()
        {
            for (int a = 0; a < ownColliders.Count; a++)
                for (int b = a + 1; b < ownColliders.Count; b++)
                {
                    var ra = ownColliders[a].attachedRigidbody;
                    var rb = ownColliders[b].attachedRigidbody;
                    if (ra == Torso || rb == Torso ||
                        (ra.TryGetComponent(out ConfigurableJoint ja) && ja.connectedBody == rb) ||
                        (rb.TryGetComponent(out ConfigurableJoint jb) && jb.connectedBody == ra))
                        Physics.IgnoreCollision(ownColliders[a], ownColliders[b]);
                }
        }

        public void SetController(IHorseController replacement)
        {
            controller = replacement ?? new StandingPoseController();
        }

        public void FixedStep(IReadOnlyList<HorseActor> opponents, bool running)
        {
            if (Torso == null || !running || retired) return;
            if (!ExternalActionMode && physicsTick % 5 == 0 && IsPolicyActive)
            {
                HorseAction action = controller.Decide(BuildObservation(opponents));
                if (action == null || !action.IsValid()) Debug.LogError("Invalid horse action: " + Id);
                else { lastAction = action; ApplyAction(action); }
            }
            physicsTick++;
            Lidar.Step(this);
            UpdateFall();
            for (int i = 0; i < 4; i++)
            {
                hoofContact[i] = hoofGroundContacts[i].Count > 0;
                Vector3 slip=hooves[i].linearVelocity;
                slip.y=0f;
                if(hoofContact[i] && Config.hoofSmoothness>.35f && slip.magnitude>1f && Time.time>=nextSlipEvent)
                {
                    nextSlipEvent=Time.time+.2f;
                    Emit(HorseEventType.Slip,hooves[i].position,slip.magnitude);
                }
            }
            Gait.Step(this,Time.fixedDeltaTime,GaitTask);
        }
        public bool SetExternalAction(HorseAction action)
        {
            if (!ExternalActionMode || !IsPolicyActive || action == null || !action.IsValid()) return false;
            lastAction = action;
            return ApplyAction(action);
        }

        public bool ApplyAction(HorseAction action)
        {
            if (action == null || !action.IsValid()) return false;
            foreach (var motor in motors)
            {
                float primary = action.values[motor.angleIndex];
                float side = motor.root ? action.values[motor.angleIndex + 1] : 0f;
                motor.descriptor.Drive(motor.descriptor.ActionAngle(primary),side);
                float power = (action.values[HorseAction.AngleCount + motor.strengthIndex] + 1f) * .5f;
                if(Condition==HorseCondition.Fallen)power*=.25f;
                var drive = motor.joint.slerpDrive;
                drive.maximumForce = rig.rootDrive * (motor.back ? Config.backStrength : Config.frontStrength) * power;
                motor.joint.slerpDrive = drive;
            }
            return true;
        }

        public HorseObservation BuildObservation(IReadOnlyList<HorseActor> opponents)
        {
            var o = new HorseObservation();
            o.Add(Torso.transform.InverseTransformDirection(Torso.linearVelocity) / 15f);
            o.Add(Torso.transform.InverseTransformDirection(Torso.angularVelocity) / 15f);
            o.Add(Torso.transform.InverseTransformDirection(Vector3.down));
            o.Add(Vector3.forward);
            foreach (var joint in allJoints)
            {
                Quaternion relative = Quaternion.Inverse(joint.connectedBody.rotation) * joint.transform.rotation;
                Vector3 right = relative * Vector3.right, up = relative * Vector3.up;
                o.Add(right); o.Add(up);
                o.Add((joint.GetComponent<Rigidbody>().angularVelocity - joint.connectedBody.angularVelocity) / 15f);
            }
            for (int i=0;i<4;i++) { o.Add(hoofContact[i] ? 1f : 0f); o.Add(hooves[i].linearVelocity / 15f); }
            o.Add(Config.bulk); o.Add(Config.frontLength); o.Add(Config.backLength); o.Add(Config.bodyLength);
            o.Add(Config.frontStrength); o.Add(Config.backStrength); o.Add(Config.massKg/750f); o.Add(Config.hoofSmoothness);
            o.Add(Torso.position.z / 100f); o.Add(Torso.position.x / 6f);
            o.Add((Torso.position.x + 6f) / 12f); o.Add((6f-Torso.position.x)/12f);
            int seen=0;
            foreach (var opponent in opponents)
            {
                if (opponent == this) continue;
                o.Add((opponent.Torso.position-Torso.position) / 20f);
                o.Add((opponent.Torso.linearVelocity-Torso.linearVelocity) / 15f);
                o.Add(opponent.Condition == HorseCondition.Fallen ? 1f : 0f);
                o.Add(opponent.Condition == HorseCondition.Recovering ? 0f : 1f);
                seen++;
            }
            for (;seen<3;seen++) for(int i=0;i<8;i++) o.Add(0f);
            foreach (float value in lastAction.values) o.Add(value);
            if(o.values.Count!=HorseObservation.Size)
                Debug.LogError("Observation layout mismatch: "+o.values.Count);
            return o;
        }

        void UpdateFall()
        {
            bool leaning = Vector3.Angle(Torso.transform.up, Vector3.up) > 65f;
            bool collapsed = Torso.position.y < restTorsoHeight * .55f;
            bool unstablePose = leaning || collapsed || torsoOnGround;
            if (torsoOnGround) groundContactTimer += Time.fixedDeltaTime;
            else groundContactTimer = 0f;
            conditionTimer += Time.fixedDeltaTime;
            if (Condition == HorseCondition.Standing && unstablePose)
            { Condition = HorseCondition.Unstable; conditionTimer = 0f; }
            else if (Condition == HorseCondition.Unstable)
            {
                if (!unstablePose) { Condition = HorseCondition.Standing; conditionTimer = 0f; }
                else if (conditionTimer >= .35f || groundContactTimer >= .2f)
                { Condition = HorseCondition.Fallen; conditionTimer = 0f; fallPosition=Torso.position;
                    foreach(var motor in motors)
                    {
                        var drive=motor.joint.slerpDrive;drive.maximumForce=0f;motor.joint.slerpDrive=drive;
                    }
                    Emit(HorseEventType.Fall, new Vector3(Torso.position.x,.03f,Torso.position.z), 1f); }
            }
            else if (Condition == HorseCondition.Fallen && assistRecovery && conditionTimer >= 3f)
                TryRecover();
            else if (Condition == HorseCondition.Recovering && conditionTimer >= Time.fixedDeltaTime)
            {
                foreach (var body in bodies) body.isKinematic = false;
                Condition = HorseCondition.Standing; conditionTimer = 0f;
            }
        }

        void TryRecover()
        {
            var physicsScene=gameObject.scene.GetPhysicsScene();
            var groundHits=new RaycastHit[32];
            int groundCount=physicsScene.Raycast(fallPosition+Vector3.up*5f,Vector3.down,
                groundHits,10f,Physics.DefaultRaycastLayers,QueryTriggerInteraction.Ignore);
            float groundHeight=float.NegativeInfinity;
            for(int i=0;i<groundCount;i++)
                if(groundHits[i].collider.GetComponentInParent<HorseGroundSurface>()!=null &&
                    groundHits[i].point.y>groundHeight)
                    groundHeight=groundHits[i].point.y;
            if(float.IsNegativeInfinity(groundHeight))return;
            Vector3 query=new Vector3(fallPosition.x,groundHeight+restTorsoClearance,fallPosition.z);
            bool clear = true;
            int hitCount=physicsScene.OverlapBox(query,new Vector3(.65f,.65f,1.4f),
                recoveryHits,Quaternion.identity,Physics.DefaultRaycastLayers,QueryTriggerInteraction.Ignore);
            if(hitCount==recoveryHits.Length)return;
            for(int hitIndex=0;hitIndex<hitCount;hitIndex++)
            {
                Collider hit=recoveryHits[hitIndex];
                var surface=hit.GetComponentInParent<HorseGroundSurface>();
                if (hit.GetComponentInParent<HorseActor>() != this &&
                    (surface==null || !surface.permitsRecovery))
                { clear = false; break; }
            }
            if (!clear) return;
            {
                for(int i=0;i<bodies.Count;i++)
                {
                    var body=bodies[i];
                    body.linearVelocity=Vector3.zero;
                    body.angularVelocity=Vector3.zero;
                    body.isKinematic=true;
                    body.position=query+restOffsets[i];
                    body.rotation=restRotations[i];
                }
                Physics.SyncTransforms();
                torsoOnGround=false;
                torsoGroundContacts.Clear();
                foreach (var contacts in hoofGroundContacts) contacts.Clear();
                Array.Clear(hoofContact,0,hoofContact.Length);
                groundContactTimer=0f;
                Condition = HorseCondition.Recovering; conditionTimer = 0f;
                lastAction=new HorseAction();
                for(int i=HorseAction.AngleCount;i<HorseAction.Size;i++)lastAction.values[i]=.7f;
                ApplyAction(lastAction);
                Lidar.Reset(this);
                Emit(HorseEventType.Recovery, query, 1f);
            }
        }

        public void Contact(Collision collision, Collider selfCollider)
        {
            if (collision.collider.GetComponentInParent<HorseActor>() == this) return;
            bool ground = collision.collider.GetComponentInParent<HorseGroundSurface>() != null;
            if (ground && selfCollider.attachedRigidbody == Torso)
            { torsoGroundContacts.Add(collision.collider); torsoOnGround = true; }
            if (ground) for(int i=0;i<hooves.Length;i++)
                if(selfCollider.attachedRigidbody==hooves[i]) hoofGroundContacts[i].Add(collision.collider);
            if (!ground && Time.time-lastCollisionEvent>.12f && collision.impulse.magnitude > Config.massKg*.25f)
            {
                lastCollisionEvent = Time.time;
                Emit(HorseEventType.Collision, collision.GetContact(0).point,
                    collision.impulse.magnitude / Config.massKg);
            }
        }
        public void ExitContact(Collision collision, Collider selfCollider)
        {
            if (selfCollider.attachedRigidbody == Torso)
            { torsoGroundContacts.Remove(collision.collider); torsoOnGround=torsoGroundContacts.Count>0; }
            for(int i=0;i<hooves.Length;i++)
                if(selfCollider.attachedRigidbody==hooves[i]) hoofGroundContacts[i].Remove(collision.collider);
        }
        void Emit(HorseEventType type, Vector3 point, float intensity)
            => EventRaised?.Invoke(new HorseEvent { type=type, horseId=Id, point=point, intensity=intensity, time=Time.time });

        public void Push(Vector3 direction, float metersPerSecond)
        {
            Torso.AddForce(direction.normalized * Torso.mass * metersPerSecond, ForceMode.Impulse);
            Emit(HorseEventType.TestImpulse, Torso.position, metersPerSecond);
        }
        public void RetireFromRace()
        {
            retired=true;
            foreach (var body in bodies)
            {
                body.linearVelocity=Vector3.zero;
                body.angularVelocity=Vector3.zero;
                body.isKinematic=true;
                foreach (var collider in body.GetComponents<Collider>()) collider.enabled=false;
            }
        }

        void LateUpdate()
        {
            if (Torso == null || trainingMode) return;
            foreach (var item in follow)
            {
                item.bone.position = item.body.transform.TransformPoint(item.positionOffset);
                item.bone.rotation = item.body.rotation * item.rotationOffset;
            }
            float sway=Mathf.Sin(Time.unscaledTime*3f+Id)*7f;
            if(bones.TryGetValue("Pelvis.002",out var tail))tail.Rotate(0f,sway,0f,Space.Self);
            if(bones.TryGetValue("Ear.L",out var earL))earL.Rotate(sway*.25f,0f,0f,Space.Self);
            if(bones.TryGetValue("Ear.R",out var earR))earR.Rotate(-sway*.25f,0f,0f,Space.Self);
        }
        void OnDestroy()
        {
            Lidar.Dispose();
            if(uniqueMesh!=null)Destroy(uniqueMesh);
            if(uniqueBodyMaterial!=null)Destroy(uniqueBodyMaterial);
            if(uniqueOutlineMaterial!=null)Destroy(uniqueOutlineMaterial);
            if(uniqueHoofMaterial!=null)Destroy(uniqueHoofMaterial);
        }
    }

    public sealed class HorseContactRelay : MonoBehaviour
    {
        public HorseActor Owner;
        void OnCollisionEnter(Collision collision) => Owner.Contact(collision, GetComponent<Collider>());
        void OnCollisionStay(Collision collision) => Owner.Contact(collision, GetComponent<Collider>());
        void OnCollisionExit(Collision collision) => Owner.ExitContact(collision, GetComponent<Collider>());
    }
}

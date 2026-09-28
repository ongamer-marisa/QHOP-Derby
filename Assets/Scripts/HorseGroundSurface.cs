using UnityEngine;

namespace QHop
{
    // Ground identification is explicit; visual names are not part of the physics contract.
    public sealed class HorseGroundSurface : MonoBehaviour
    {
        public bool permitsRecovery = true;
    }
}

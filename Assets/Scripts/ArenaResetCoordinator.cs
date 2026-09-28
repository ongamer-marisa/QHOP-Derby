using UnityEngine;

namespace QHop
{
    public sealed class ArenaResetCoordinator : MonoBehaviour
    {
        public ArenaController arena;
        public void RequestReset()
        {
            if(arena==null)arena=GetComponent<ArenaController>();
            if(arena!=null)arena.QueueReset();
        }
    }
}

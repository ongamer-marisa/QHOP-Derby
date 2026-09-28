using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace QHop
{
    public sealed class HorseTelemetryRecorder : MonoBehaviour
    {
        const int MaxRows=12000;
        readonly Queue<string> rows=new Queue<string>();
        float nextSample;
        public void Sample(IReadOnlyList<HorseActor> horses)
        {
            if(Time.time<nextSample)return;
            nextSample=Time.time+.05f;
            foreach(var horse in horses)
            {
                var p=horse.Torso.position;var v=horse.Torso.linearVelocity;
                Add(string.Join(",",Time.time.ToString("F3",CultureInfo.InvariantCulture),horse.Id,
                    horse.Condition,p.x.ToString("F3",CultureInfo.InvariantCulture),p.y.ToString("F3",CultureInfo.InvariantCulture),
                    p.z.ToString("F3",CultureInfo.InvariantCulture),v.x.ToString("F3",CultureInfo.InvariantCulture),
                    v.y.ToString("F3",CultureInfo.InvariantCulture),v.z.ToString("F3",CultureInfo.InvariantCulture),""));
            }
        }
        public void Record(HorseEvent value)
        {
            Add(string.Join(",",value.time.ToString("F3",CultureInfo.InvariantCulture),value.horseId,
                "",value.point.x.ToString("F3",CultureInfo.InvariantCulture),
                value.point.y.ToString("F3",CultureInfo.InvariantCulture),
                value.point.z.ToString("F3",CultureInfo.InvariantCulture),"","","",value.type));
        }
        void Add(string row)
        {
            rows.Enqueue(row);while(rows.Count>MaxRows)rows.Dequeue();
        }
        public string ExportCsv()
        {
            string path=Path.Combine(Application.persistentDataPath,"horse_telemetry.csv");
            var text=new StringBuilder("time,horse_id,condition,x,y,z,vx,vy,vz,event\n");
            foreach(var row in rows)text.AppendLine(row);
            File.WriteAllText(path,text.ToString(),Encoding.UTF8);
            return path;
        }
    }
}

"""Per-episode event memory; peeking values never commits a scan."""
import numpy as np
import torch
from .schema import SLICES, VISION_HIDDEN


class VisionStates:
    def __init__(self, model, device):
        self.model=model;self.device=device;self.entries={};self.serial=0

    def entry(self,key):
        if key not in self.entries:
            self.serial+=1
            self.entries[key]={'scan_id':None,'scans':[], 'hidden':torch.zeros(1,VISION_HIDDEN,device=self.device),'serial':self.serial,'updates':0}
        return self.entries[key]

    @torch.no_grad()
    def consume(self,key,raw,commit=True):
        e=self.entry(key);scan_id=int(raw[SLICES.scan_id])
        new=scan_id!=e['scan_id']
        if e['scan_id'] is not None and scan_id<e['scan_id']:
            raise ValueError('Scan sequence regressed without an episode reset')
        prefix=tuple(e['scans'])
        hidden=e['hidden']
        if new:
            scan=np.array(raw[SLICES.lidar],dtype=np.float32,copy=True)
            hidden=self.model(torch.as_tensor(scan,device=self.device)[None],hidden)
            if commit:
                e['scans'].append(scan);e['hidden']=hidden;e['scan_id']=scan_id;e['updates']+=1
        return hidden,new,prefix,e['serial']

    def terminal(self,key):self.entries.pop(key,None)
    def clear(self):self.entries.clear()

    @torch.no_grad()
    def refresh(self):
        for e in self.entries.values():
            h=torch.zeros(1,VISION_HIDDEN,device=self.device)
            for scan in e['scans']:h=self.model(torch.as_tensor(scan,device=self.device)[None],h)
            e['hidden']=h

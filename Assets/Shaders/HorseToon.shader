Shader "QHOP/SoftToon"
{
    Properties
    {
        _BaseColor("Base Color", Color) = (0.7,0.4,0.25,1)
        _ShadowColor("Shadow Color", Color) = (0.48,0.57,0.63,1)
        _RimColor("Rim Color", Color) = (1,0.88,0.66,1)
        _StepSoftness("Step Softness", Range(0.01,0.5)) = 0.18
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        Pass
        {
            Name "Toon Forward"
            Tags { "LightMode"="UniversalForward" }
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            CBUFFER_START(UnityPerMaterial)
            half4 _BaseColor;
            half4 _ShadowColor;
            half4 _RimColor;
            half _StepSoftness;
            CBUFFER_END
            struct Attributes { float4 positionOS:POSITION; float3 normalOS:NORMAL; };
            struct Varyings { float4 positionCS:SV_POSITION; float3 positionWS:TEXCOORD0; half3 normalWS:TEXCOORD1; float4 shadowCoord:TEXCOORD2; };
            Varyings vert(Attributes i)
            {
                Varyings o;
                o.positionWS=TransformObjectToWorld(i.positionOS.xyz);
                o.positionCS=TransformWorldToHClip(o.positionWS);
                o.normalWS=TransformObjectToWorldNormal(i.normalOS);
                o.shadowCoord=TransformWorldToShadowCoord(o.positionWS);
                return o;
            }
            half4 frag(Varyings i):SV_Target
            {
                Light light=GetMainLight(i.shadowCoord);
                half3 n=normalize(i.normalWS);
                half diffuse=saturate(dot(n,light.direction)) * light.shadowAttenuation;
                half band=smoothstep(.28-_StepSoftness,.28+_StepSoftness,diffuse)*.65+
                          smoothstep(.72-_StepSoftness,.72+_StepSoftness,diffuse)*.35;
                half3 shadow=_BaseColor.rgb*_ShadowColor.rgb;
                half3 color=lerp(shadow,_BaseColor.rgb,band);
                half rim=pow(1-saturate(dot(n,normalize(_WorldSpaceCameraPos-i.positionWS))),3);
                color+=_RimColor.rgb*rim*.13;
                return half4(color,1);
            }
            ENDHLSL
        }
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode"="ShadowCaster" }
            ZWrite On
            HLSLPROGRAM
            #pragma vertex shadowVert
            #pragma fragment shadowFrag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            struct Varyings { float4 positionCS:SV_POSITION; };
            Varyings shadowVert(float4 positionOS:POSITION)
            {
                Varyings o; o.positionCS=TransformWorldToHClip(TransformObjectToWorld(positionOS.xyz)); return o;
            }
            half4 shadowFrag(Varyings i):SV_Target { return 0; }
            ENDHLSL
        }
    }
}

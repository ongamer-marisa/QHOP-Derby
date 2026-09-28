Shader "QHOP/Outline"
{
    Properties
    {
        _OutlineColor("Outline Color", Color)=(0.14,0.18,0.20,1)
        _Width("World Width", Range(0.001,0.08))=0.018
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        Pass
        {
            Name "Horse Outline"
            Tags { "LightMode"="UniversalForward" }
            Cull Front
            ZWrite On
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            CBUFFER_START(UnityPerMaterial)
            half4 _OutlineColor;
            float _Width;
            CBUFFER_END
            struct Attributes { float4 positionOS:POSITION;float3 normalOS:NORMAL; };
            struct Varyings {float4 positionCS:SV_POSITION;};
            Varyings vert(Attributes i)
            {
                Varyings o;
                float3 world=TransformObjectToWorld(i.positionOS.xyz);
                world+=normalize(TransformObjectToWorldNormal(i.normalOS))*_Width;
                o.positionCS=TransformWorldToHClip(world);
                return o;
            }
            half4 frag(Varyings i):SV_Target{return _OutlineColor;}
            ENDHLSL
        }
    }
}

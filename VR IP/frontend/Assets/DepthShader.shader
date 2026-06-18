Shader "Custom/DepthGrayscale"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata { float4 vertex : POSITION; };
            struct v2f { float4 pos : SV_POSITION; float depth : TEXCOORD0; };

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                // COMPUTE_EYEDEPTH gives depth in meters from the camera
                o.depth = -UnityObjectToViewPos(v.vertex).z;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // Convert depth (0 to 5 meters) to 0-1 grayscale
                // Adjust "5.0" to change the max distance
                float d = i.depth / 5.0; 
                return fixed4(d, d, d, 1);
            }
            ENDCG
        }
    }
}
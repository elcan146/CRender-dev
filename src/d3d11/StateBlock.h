// StateBlock — snapshot of every pipeline slot our Present-time passes touch.
// Get* calls AddRef; Restore re-binds and releases. Shared by RetroPass and AiPass.
#pragma once

#include <d3d11.h>

namespace retro
{

struct StateBlock
{
    ID3D11InputLayout*        il = nullptr;
    D3D11_PRIMITIVE_TOPOLOGY  topo = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
    ID3D11VertexShader*       vs = nullptr;
    ID3D11ClassInstance*      vsCI[256] = {};
    UINT                      vsCIn = 256;
    ID3D11PixelShader*        ps = nullptr;
    ID3D11ClassInstance*      psCI[256] = {};
    UINT                      psCIn = 256;
    ID3D11GeometryShader*     gs = nullptr;
    ID3D11HullShader*         hs = nullptr;
    ID3D11DomainShader*       ds = nullptr;
    ID3D11ShaderResourceView* psSRV[2] = {};
    ID3D11SamplerState*       psSamp[2] = {};
    ID3D11Buffer*             psCB = nullptr;
    ID3D11ShaderResourceView* vsSRV = nullptr; // HUD instance buffer slot
    ID3D11Buffer*             vsCB = nullptr;
    ID3D11RenderTargetView*   rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView*   dsv = nullptr;
    ID3D11BlendState*         blend = nullptr;
    FLOAT                     blendF[4] = {};
    UINT                      blendMask = 0xFFFFFFFF;
    ID3D11DepthStencilState*  dss = nullptr;
    UINT                      stencilRef = 0;
    ID3D11RasterizerState*    rs = nullptr;
    D3D11_VIEWPORT            vp[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
    UINT                      vpN = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;

    void Capture(ID3D11DeviceContext* c) noexcept
    {
        c->IAGetInputLayout(&il);
        c->IAGetPrimitiveTopology(&topo);
        c->VSGetShader(&vs, vsCI, &vsCIn);
        c->PSGetShader(&ps, psCI, &psCIn);
        c->GSGetShader(&gs, nullptr, nullptr);
        c->HSGetShader(&hs, nullptr, nullptr);
        c->DSGetShader(&ds, nullptr, nullptr);
        c->PSGetShaderResources(0, 2, psSRV);
        c->PSGetSamplers(0, 2, psSamp);
        c->PSGetConstantBuffers(0, 1, &psCB);
        c->VSGetShaderResources(0, 1, &vsSRV);
        c->VSGetConstantBuffers(0, 1, &vsCB);
        c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, &dsv);
        c->OMGetBlendState(&blend, blendF, &blendMask);
        c->OMGetDepthStencilState(&dss, &stencilRef);
        c->RSGetState(&rs);
        c->RSGetViewports(&vpN, vp);
    }

    void Restore(ID3D11DeviceContext* c) noexcept
    {
        c->IASetInputLayout(il);
        c->IASetPrimitiveTopology(topo);
        c->VSSetShader(vs, vsCI, vsCIn);
        c->PSSetShader(ps, psCI, psCIn);
        c->GSSetShader(gs, nullptr, 0);
        c->HSSetShader(hs, nullptr, 0);
        c->DSSetShader(ds, nullptr, 0);
        c->PSSetShaderResources(0, 2, psSRV);
        c->PSSetSamplers(0, 2, psSamp);
        c->PSSetConstantBuffers(0, 1, &psCB);
        c->VSSetShaderResources(0, 1, &vsSRV);
        c->VSSetConstantBuffers(0, 1, &vsCB);
        c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, dsv);
        c->OMSetBlendState(blend, blendF, blendMask);
        c->OMSetDepthStencilState(dss, stencilRef);
        c->RSSetState(rs);
        c->RSSetViewports(vpN, vp);

        auto rel = [](IUnknown* p) { if (p != nullptr) p->Release(); };
        rel(il); rel(vs); rel(ps); rel(gs); rel(hs); rel(ds);
        for (UINT i = 0; i < vsCIn; ++i) rel(vsCI[i]);
        for (UINT i = 0; i < psCIn; ++i) rel(psCI[i]);
        rel(psSRV[0]); rel(psSRV[1]); rel(psSamp[0]); rel(psSamp[1]); rel(psCB);
        rel(vsSRV); rel(vsCB);
        for (auto* r : rtv) rel(r);
        rel(dsv); rel(blend); rel(dss); rel(rs);
    }
};

} // namespace retro

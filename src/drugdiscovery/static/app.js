"use strict";
/* Drug Discovery Platform — single-page dashboard logic. */
const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => [...r.querySelectorAll(s)];
function esc(s){ const d=document.createElement("div"); d.textContent = s==null?"":String(s); return d.innerHTML; }

let state = { ws: null, workspaces: [], view: "overview", timers: [] };

const KIND_COLOR = {
  disease:"#e5646e", phenotype:"#e5b567", tissue:"#b48ead", cell_type:"#9d7cd8",
  biological_process:"#37d399", pathway:"#37d399", gene:"#4f8cff", protein:"#4f8cff",
  receptor:"#4f8cff", enzyme:"#7aa2ff", transcription_factor:"#7aa2ff",
  hormone:"#e5b567", metabolite:"#e5b567", ligand:"#e5b567",
  compound:"#c678dd", drug:"#c678dd", drug_class:"#c678dd", binding_site:"#7f9cf5",
  interaction:"#a0a8c0", publication:"#7c87a6", experiment:"#a0a8c0",
  simulation:"#a0a8c0", question:"#e5b567", other:"#7c87a6"
};
const EL_COLOR = { established:"#37d399", probable:"#7aa2ff", speculative:"#e5b567", computational:"#5a6480" };

async function api(path, opts={}) {
  const r = await fetch("/api"+path, { headers: {"Content-Type":"application/json"}, ...opts });
  if (!r.ok) { const t = await r.text().catch(()=>""); throw new Error(r.status+" "+t.slice(0,120)); }
  return r.json();
}

function confBar(conf){ const c = (conf||0)*100; return `<div class="conf-bar"><div class="conf-fill" style="width:${c}%"></div></div>`; }
function badge(lvl){ return `<span class="badge ${esc(lvl)}">${esc(lvl)}</span>`; }

/* ---------------- workspace/research ---------------- */
async function refreshWorkspaces(){
  state.workspaces = await api("/workspaces");
  const sel = $("#ws-select");
  const wrap = $("#ws-select-wrap");
  if (!state.workspaces.length){
    wrap.innerHTML = `<span class="muted">No workspaces yet — run: <span class="mono">uv run python -m drugdiscovery.cli seed</span></span>`;
    return;
  }
  if (!sel){
    wrap.innerHTML = `<span class="muted">workspace</span> <select id="ws-select"></select>`;
  }
  const s2 = $("#ws-select");
  const cur = state.ws?.id || state.workspaces[0].id;
  s2.innerHTML = state.workspaces.map(w=>`<option value="${w.id}" ${w.id===cur?"selected":""}>${esc(w.name)}</option>`).join("");
  s2.onchange = async () => { state.ws = state.workspaces.find(w=>w.id===s2.value); await refresh(); };
  state.ws = state.workspaces.find(w=>w.id===cur) || state.workspaces[0];
  await refreshResearchState();
}

async function refreshResearchState(){
  if (!state.ws) return;
  const w = await api("/workspaces/"+state.ws.id);
  const el = $("#research-state");
  const st = w.research_state;
  el.className = "pill pill-"+st;
  el.textContent = st==="running"?"● running":(st==="paused"?"‖ paused":(st==="stopped"?"■ stopped":"— "+st));
}

function researchAction(action){ return async ()=>{ await api(`/workspaces/${state.ws.id}/research`, {method:"POST", body:JSON.stringify({action})}); await refreshWorkspaces(); await render(); }; }
function bindResearch(){
  $("#btn-start").onclick = researchAction("start");
  $("#btn-pause").onclick = researchAction("pause");
  $("#btn-cycle").onclick = researchAction("cycle");
  $("#btn-search").onclick = ()=> $("#search-modal").classList.remove("hidden");
  $("#search-cancel").onclick = ()=> $("#search-modal").classList.add("hidden");
  $("#search-go").onclick = async ()=>{
    const query = $("#search-q").value.trim(); if(!query) return;
    const size = +$("#search-size").value || 10;
    const provider = $("#search-provider").value;
    const out = $("#search-result");
    out.textContent = "searching…";
    try {
      const r = await api(`/workspaces/${state.ws.id}/search`, {method:"POST", body:JSON.stringify({query,size,provider})});
      out.textContent = `ingested ${r.new} new / ${r.seen} seen`;
      setTimeout(()=> $("#search-modal").classList.add("hidden"), 1200);
      await render();
    } catch(e){ out.textContent = "error: "+e.message; }
  };
}

/* ---------------- views ---------------- */
function render(){
  const m = $("#main"); m.innerHTML = `<div class="placeholder">Loading…</div>`;
  return renderView({view:state.view}).then(html=>{ m.innerHTML = html; bindView(state.view); });
}

function renderView({view}){
  if (view==="overview") return renderOverview();
  if (view==="map") return renderMapPromise();
  if (view==="targets") return renderTargets();
  if (view==="compounds") return renderCompounds();
  if (view==="structures") return renderStructures();
  if (view==="simulations") return renderSimulations();
  if (view==="candidates") return renderCandidates();
  if (view==="literature") return renderLiterature();
  if (view==="hypotheses") return renderHypotheses();
  if (view==="queue") return renderQueue();
  if (view==="timeline") return renderTimeline();
  if (view==="system") return renderSystem();
  return Promise.resolve("<div class='placeholder'>unknown</div>");
}

async function renderOverview(){
  if (!state.ws) return "<div class='placeholder'>No workspace selected.</div>";
  const o = await api(`/workspaces/${state.ws.id}/overview`);
  const c = o.counts, k = c.by_kind||{};
  const kinds = Object.entries(k).sort((a,b)=>b[1]-a[1]).slice(0,10)
    .map(([kk,n])=>`<span class="pill">${kk} ${n}</span>`).join(" ");
  const hyp = (o.top_hypotheses||[]).map(h=>`<div class="card">
     <h4>${esc(h.title)}</h4>
     <div class="small muted">status: ${esc(h.status)}</div>
     ${confBar(h.confidence)}</div>`).join("");
  const tl = (o.recent_timeline||[]).slice(0,6).map(t=>`<div class="small" style="margin:3px 0"><span class="mono muted">${esc(t.ts.slice(11,19))}</span> · <span class="muted">${esc(t.agent)}</span> — ${esc(t.message)}</div>`).join("");
  return `<h1>${esc(o.name)}</h1>
  <div class="sub">${esc(o.disease||"no disease bound")} · research: <b>${esc(o.research_state)}</b></div>
  <div class="stat-grid">
    <div class="card"><div class="k">Graph nodes</div><div class="big">${c.nodes}</div></div>
    <div class="card"><div class="k">Edges</div><div class="big">${c.edges}</div></div>
    <div class="card"><div class="k">Papers</div><div class="big">${c.papers}</div></div>
    <div class="card"><div class="k">Claims</div><div class="big">${c.statements}</div></div>
    <div class="card"><div class="k">Hypotheses</div><div class="big">${c.hypotheses}</div></div>
    <div class="card"><div class="k">Queued experiments</div><div class="big">${c.queued_experiments}</div></div>
  </div>
  <div class="cards" style="margin-top:14px"><div class="card" style="grid-column:1/-1"><div class="k">Graph composition</div><div style="margin-top:8px">${kinds}</div></div></div>
  <h3 style="margin-top:18px">Leading hypotheses</h3>
  <div class="cards">${hyp || '<div class="card muted">no hypotheses yet — start research or run a cycle</div>'}</div>
  <h3 style="margin-top:18px">Recent research activity</h3>
  <div class="card" style="max-width:860px">${tl||"<span class='muted'>no activity yet</span>"}</div>`;
}

async function renderMapPromise(){
  if (!state.ws) return "<div class='placeholder'>No workspace.</div>";
  const graph = await api(`/workspaces/${state.ws.id}/graph`);
  if (!graph.nodes.length) return "<div class='placeholder'>Knowledge graph is empty — ingest literature first.</div>";
  return `<h1>Mechanism Map</h1><div class="sub">Interactive knowledge graph · click a node for details</div><div id="graph-host"></div>
  <div class="sub small" style="margin-top:8px">Colors: ${Object.entries(KIND_COLOR).slice(0,14).map(([k,c])=>`<span class="badge" style="color:${c};border-color:${c}">${k}</span>`).join(" ")}</div>`;
}

async function openNodeCard(nid){
  const modal=document.getElementById("node-modal"); modal.hidden=false;
  const body=document.getElementById("node-body");
  body.innerHTML="<div class='muted'>Loading research card…</div>";
  try{
    const c=await api(`/nodes/${nid}`);
    document.getElementById("node-title").textContent=c.name+" ("+c.kind+")";
    const prov=e=>(e.provenance&&e.provenance.length)?e.provenance.map(p=>p.paper_id?p.paper_id.slice(0,8):(p.kind||"prov")).join(", "):"—";
    const edges=c.edges.map(e=>"<tr><td>"+(e.direction==="out"?"→":"←")+" <b>"+esc(e.neighbor)+"</b> <span class='badge'>"+esc(e.neighbor_kind)+"</span></td><td class='mono'>"+esc(e.type)+"</td><td>"+badge(e.evidence_level)+"</td><td class='small'>"+esc(e.knowledge_type)+"</td><td class='small mono'>"+esc(prov(e))+"</td></tr>").join("");
    const stmts=c.statements.map(s=>"<div class='stmt'><span class='badge'>"+esc(s.claim_type)+"</span> "+esc(s.text)+"</div>").join("");
    const papers=c.papers.map(p=>"<div class='paperline'><b>"+esc(p.title)+"</b> <span class='muted small'>"+esc(p.journal||"")+" "+(p.pmid?("PMID:"+p.pmid):"")+" "+(p.doi?("DOI:"+p.doi):"")+"</span></div>").join("");
    let html="";
    html+="<div class='kvrow'><span class='k'>evidence</span>"+badge(c.evidence_level)+" <span class='k'>knowledge_type</span><span class='mono'>"+esc(c.knowledge_type)+"</span> <span class='k'>conf</span>"+c.confidence+"</div>";
    if(c.summary) html+="<div class='stmt'>"+esc(c.summary)+"</div>";
    if(c.synonyms&&c.synonyms.length) html+="<div class='small muted'>synonyms: "+c.synonyms.map(esc).join(", ")+"</div>";
    if(c.compound) html+="<div class='kvrow'><span class='k'>compound</span><span class='mono'>"+esc(c.compound.name)+"</span> <span class='badge'>"+esc(c.compound.status)+"</span></div>";
    if(c.structures.length) html+="<div class='kvrow'><span class='k'>structures</span>"+c.structures.map(s=>"<span class='badge'>"+esc(s.source_id)+" ("+esc(s.source)+")</span>").join(" ")+"</div>";
    html+="<h3>Edges ("+c.edges.length+")</h3><table><thead><tr><th>Neighbor</th><th>Type</th><th>Level</th><th>Knowledge</th><th>Provenance</th></tr></thead><tbody>"+(edges||"<tr><td colspan=5 class=muted>none</td></tr>")+"</tbody></table>";
    html+="<h3>Statements ("+c.statements.length+")</h3>"+(stmts||"<div class='muted'>none</div>");
    html+="<h3>Papers ("+c.papers.length+")</h3>"+(papers||"<div class='muted'>none</div>");
    body.innerHTML=html;
  }catch(err){ body.innerHTML="<div class='muted'>card error: "+esc(String(err.message||err))+"</div>"; }
}

const cmpSel = new Set();
function toggleCmp(el){ if(el.checked) cmpSel.add(el.dataset.cid); else cmpSel.delete(el.dataset.cid); }
async function openCompoundCard(cid){
  const modal=document.getElementById("cmp-modal"); modal.hidden=false;
  const body=document.getElementById("cmp-body");
  body.innerHTML="<div class='muted'>Loading compound card…</div>";
  try{
    const c=await api(`/compounds/${cid}`);
    const sim=await api(`/compounds/${cid}/similar?min_sim=0.3`).catch(()=>[]);
    let sims=[];
    if(state.ws){ const all=await api(`/workspaces/${state.ws.id}/simulations`).catch(()=>[]);
      sims=(Array.isArray(all)?all:[]).filter(r=>r.compound_id===cid); }
    document.getElementById("cmp-title").textContent=c.name;
    const dockRows=sims.map(r=>`<tr><td class="mono">${esc(r.kind)}</td><td>${esc(r.structure_source_id||r.structure_id.slice(0,8))}</td><td class="mono">${r.score!=null?r.score.toFixed(2):"—"}</td><td>${badgeCls2(r.status)}</td><td class="small mono">${esc(r.software_version||r.software||"")} seed ${r.seed}</td><td class="small muted">${esc((r.assumptions||"").slice(0,80))}…</td></tr>`).join("");
    let html=`<div class="cmp-grid"><img class="struct big" src="/api/compounds/${cid}/svg" alt="${esc(c.name)}"/>
      <div class="descgrid">
        <span class="k">formula</span><span class="mono">${esc(c.formula||"—")}</span>
        <span class="k">MW</span><span class="mono">${c.molmass??"—"}</span>
        <span class="k">logP</span><span class="mono">${c.logp??"—"}</span>
        <span class="k">TPSA</span><span class="mono">${c.tpsa??"—"}</span>
        <span class="k">HBD/HBA</span><span class="mono">${c.hbd??"—"} / ${c.hba??"—"}</span>
        <span class="k">rotatable</span><span class="mono">${c.rotb??"—"}</span>
        <span class="k">InChIKey</span><span class="mono">${esc(c.inchikey||"—")}</span>
        <span class="k">status</span><span class="badge">${esc(c.research_status)}</span>
        <span class="k">sources</span><span class="small">${Object.keys(c.sources||{}).map(esc).join(", ")||"—"}</span>
      </div></div>
      <div class="small mono muted" style="word-break:break-all;margin:8px 0">${esc(c.smiles||"")}</div>
      <h3>Simulation history (${sims.length}) — predictions, not activity claims</h3>
      <table><thead><tr><th>Kind</th><th>Structure</th><th>Score</th><th>Status</th><th>Software/seed</th><th>Assumptions</th></tr></thead><tbody>${dockRows||'<tr><td colspan=6 class="muted">none yet</td></tr>'}</tbody></table>
      <h3>Similar compounds (${sim.filter(x=>x.id!==cid).length})</h3>
      <div class="cards">${sim.filter(x=>x.id!==cid).slice(0,8).map(x=>`<div class="card mini" onclick="openCompoundCard('${x.id}')">
        <img class="struct sm" src="/api/compounds/${x.id}/svg" loading="lazy"/>
        <div class="small"><b>${esc(x.name)}</b><br>Tanimoto ${x.similarity}</div></div>`).join("")||"<div class='muted small'>none ≥0.3</div>"}</div>`;
    body.innerHTML=html;
  }catch(err){ body.innerHTML="<div class='muted'>card error: "+esc(String(err.message||err))+"</div>"; }
}
function badgeCls2(s){ return badge(s==="done"?"established":(s==="failed"?"computational":"speculative")); }
async function cmpVs(){
  const ids=[...cmpSel];
  if(ids.length<2){ alert("Select at least 2 compounds via the compare checkboxes"); return; }
  const modal=document.getElementById("cmp-vs"); modal.hidden=false;
  const body=document.getElementById("cmp-vs-body");
  body.innerHTML="<div class='muted'>Loading…</div>";
  const cs=await Promise.all(ids.slice(0,4).map(id=>api(`/compounds/${id}`)));
  const rows=[["name","name"],["formula","formula"],["MW","molmass"],["logP","logp"],["TPSA","tpsa"],["HBD","hbd"],["HBA","hba"],["rotatable","rotb"],["status","research_status"]];
  body.innerHTML=`<table><thead><tr><th></th>${cs.map(c=>`<th><img class="struct sm" src="/api/compounds/${c.id}/svg"/><br>${esc(c.name)}</th>`).join("")}</tr></thead>
   <tbody>${rows.map(([lbl,k])=>`<tr><td class="k">${lbl}</td>${cs.map(c=>`<td class="mono">${esc(String(c[k]??"—"))}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function bindView(view){
  if (view==="map") initGraph();
  $$(".tgt-link").forEach(el=> el.onclick=()=>openNodeCard(el.dataset.nid));
  const nc = $("#node-close"); if (nc) nc.onclick = ()=> $("#node-modal").hidden = true;
  const cc = $("#cmp-close"); if (cc) cc.onclick = ()=> $("#cmp-modal").hidden = true;
  const cv = $("#cmp-vs-close"); if (cv) cv.onclick = ()=> $("#cmp-vs").hidden = true;
  const cb = $("#btn-cmp"); if (cb) cb.onclick = cmpVs;
  $$(".compound-card h4").forEach(h=> h.style.cursor="pointer", 0);
  $$(".compound-card h4").forEach(h=> h.onclick = ()=>{ const inp=h.closest(".compound-card").querySelector("input[data-cid]"); if(inp) openCompoundCard(inp.dataset.cid); });
  if (view==="compounds") bindCompound();
  if (view==="structures") bindStructures();
  if (view==="simulations") bindSimulations();
  if (view==="literature") bindLiterature();
}
function initGraph(){
  const host = $("#graph-host"); if (!host) return;
  api(`/workspaces/${state.ws.id}/graph`).then(g=>{
    const nodes = g.nodes.map(n=>({ id:n.id, label:n.label, color:{background: KIND_COLOR[n.kind]||"#7c87a6", border:"#000"},
      borderWidth:1, font:{color:"#d7ddee", size:12}, value:(0.5+(n.confidence||0))*2, shape:"dot", title:`${n.label}\n${n.kind}\nconf: ${n.confidence}\n${n.evidence_level}` }));
    const edges = g.edges.map(e=>({ id:e.id, from:e.source, to:e.target, label:e.type,
      arrows:"to", color:{color: EL_COLOR[e.evidence_level]||"#7c87a6", highlight:"#ffffff"},
      font:{color:"#8fa0c8", size:10}, title:`${e.type} (${e.evidence_level})` }));
    const ds = new vis.DataSet(nodes), es = new vis.DataSet(edges);
    const net = new vis.Network(host, { nodes:ds, edges:es }, {
      nodes:{scaling:{min:10,max:42}}, edges:{smooth:{type:"continuous"}},
      physics:{stabilization:{iterations:250}, solver:"forceAtlas2Based", forceAtlas2Based:{gravitationalConstant:-40,springLength:90}},
      interaction:{hover:true, tooltipDelay:120, hideEdgesOnDrag:false},
      layout:{improvedLayout:true}
    });
    net.once("stabilizationIterationsDone", ()=> net.setOptions({physics:false}));
    net.on("click", p=>{ if(p.nodes && p.nodes.length) openNodeCard(p.nodes[0]); });
  });
}

async function renderTargets(){
  if (!state.ws) return "<div class='placeholder'>No workspace.</div>";
  const ts = await api(`/workspaces/${state.ws.id}/targets`);
  if (!ts.length) return "<div class='placeholder'>No evidence-supported targets yet.</div>";
  const rows = ts.map(t=>`<tr>
    <td><b class="tgt-link" data-nid="${esc(t.id)}" style="cursor:pointer;text-decoration:underline dotted">${esc(t.name)}</b> <span class="badge">${esc(t.kind)}</span><br><span class="small muted">${esc((t.ligands||[]).join(", ") || "no ligands mapped")}</span></td>
    <td>${badge(t.evidence_level)}</td>
    <td>${esc(t.profile.independent_sources)}</td>
    <td>${esc(t.profile.replication)}</td>
    <td>${esc(t.profile.clinical_count)} / ${esc(t.profile.animal_count)} / ${esc(t.profile.in_vitro_count)}</td>
    <td style="width:140px">${confBar(t.profile.overall_confidence)}<span class="mono small muted">${t.profile.overall_confidence}</span></td>
  </tr>`).join("");
  return `<h1>Target Ranking</h1><div class="sub">Evidence-ranked molecular targets · ranks are multi-dimensional, not a single score</div>
  <table><thead><tr><th>Target</th><th>Level</th><th>Sources</th><th>Replication</th><th>Clin/Anim/InVit</th><th>Evidence</th></tr></thead><tbody>${rows}</tbody></table>`;
}

async function renderCompounds(){
  if (!state.ws) return "<div class='placeholder'>No workspace.</div>";
  const wsC = await api(`/workspaces/${state.ws.id}/compounds`);
  const cats = await api(`/compounds`);
  const wired = wsC.filter(x=>x.compound);
  const cards = wired.map(x=>{
    const c=x.compound;
    return `<div class="card compound-card">
      <img class="struct" src="/api/compounds/${c.id}/svg" alt="${esc(c.name)}" loading="lazy"
           onerror="this.style.display='none'"/>
      <div><h4>${esc(c.name)}</h4>
      <div class="small mono muted">${esc(c.formula||"")} · MW ${c.molmass??"–"} · logP ${c.logp??"–"} · TPSA ${c.tpsa??"–"}</div>
      <div class="small mono muted" style="word-break:break-all">${esc(c.inchikey||"")}</div>
      <label class="cmpchk"><input type="checkbox" data-cid="${esc(c.id)}" onchange="toggleCmp(this)"> compare</label>
      <div style="margin-top:6px">${(x.targets||[]).map(t=>`<span class="pill">→ ${esc(t.name)} <span class="muted">${esc(t.relation)}${t.pchembl?` · pChEMBL ${t.pchembl}`:""}</span></span>`).join(" ")||'<span class="muted small">no measured target link</span>'}</div>
      </div></div>`; }).join("");
  const sel = cats.map(c=>`<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("");
  return `<h1>Compound Library</h1>
  <div class="sub">Structure-aware catalogue (InChIKey dedup) · RDKit descriptors · measured target links from ChEMBL</div>
  <div class="stat-grid" style="margin-top:10px">
    <div class="card"><div class="k">Catalogue</div><div class="big">${cats.length}</div></div>
    <div class="card"><div class="k">Wired into workspace</div><div class="big">${wired.length}</div></div>
  </div>
  <div class="card" style="margin-top:14px">
    <div class="k">Tanimoto similarity search (against catalogue)</div>
    <div class="row" style="margin-top:8px">
      <label>Query structure (SMILES)
        <input id="sim-smiles" type="text" placeholder="Oc1ccccc1" style="min-width:260px"/></label>
      <label>or catalogue compound
        <select id="sim-seed"><option value="">— none —</option>${sel}</select></label>
      <button id="btn-sim" class="btn btn-primary">Search</button> <button id="btn-cmp" class="btn">Compare selected</button>
    </div>
    <div class="row" style="margin-top:6px"><label>min similarity
      <input id="sim-min" type="number" step="0.05" min="0" max="1" value="0.35" style="width:70px"/></label></div>
    <div id="sim-results" class="small muted" style="margin-top:8px"></div>
  </div>
  <h3 style="margin-top:18px">Workspace compounds</h3>
  <div class="cards" style="grid-template-columns:repeat(auto-fill,minmax(380px,1fr));margin-top:8px">
    ${cards || '<div class="card muted">No catalogued compounds linked here yet. Run a research cycle — the chemical-discovery step pulls measured ChEMBL ligands for top targets.</div>'}
  </div>`;
}
function bindCompound(){
  const q=sel=>document.querySelector(sel);
  const btn=q("#btn-sim");
  if(!btn) return;
  btn.onclick = async ()=>{
    const smiles=q("#sim-smiles").value.trim();
    const seed=q("#sim-seed").value;
    const box=q("#sim-results");
    if(!smiles && !seed){ box.innerHTML=""; return; }
    box.innerHTML="searching…";
    let cid="";
    if(seed){
      const all=await api(`/compounds`);
      const hit=all.find(c=>c.name.toLowerCase()===seed.toLowerCase());
      if(hit) cid=hit.id;
    }
    const where=cid?`/compounds/${cid}/similar`:`/compounds/${(await api(`/compounds`))[0]?.id}/similar?smiles=${encodeURIComponent(smiles)}`;
    try{
      let r=await api(where);
      const min=parseFloat(q("#sim-min").value||0);
      const rows=r.filter(x=>x.similarity>=min).map(x=>`<div class="row" style="gap:12px;align-items:center;border-bottom:1px solid #1c2230;padding:6px 0">
        <img class="struct" style="width:64px;height:48px" src="/api/compounds/${x.id}/svg" alt="" onerror="this.style.display='none'"/>
        <span>${esc(x.name)}</span><span class="mono muted">${esc(x.formula)}</span>
        <span class="badge ${x.similarity>=0.6?'established':(x.similarity>=0.4?'probable':'speculative')}">${(x.similarity*100).toFixed(0)}%</span>
      </div>`).join("")||"<div class='muted'>no matches above threshold</div>";
      box.innerHTML=rows;
    }catch(e){ box.innerHTML="<span class='muted'>error: "+esc(e.message||e)+"</span>"; }
  };
}

async function renderStructures(){
  const S = await api(`/workspaces/${state.ws.id}/structures`);
  const rows = S.length ? S.map(st=>`<div class="struct-card">
    <div class="struct-head"><span class="badge ${st.source==='pdb'?'fact':'pred'}">${st.source==='pdb'?'PDB experimental':'AlphaFold prediction'}</span>
      <b>${esc(st.source_id)}</b> \u00b7 ${esc(st.protein||'unlinked')}</div>
    <div class="meta">${st.method?esc(st.method):''}${st.resolution?' \u00b7 '+st.resolution+' \u00c5':''} \u00b7 ${st.ligand_count} ligand(s)</div>
    ${st.ligands&&st.ligands.length?`<div class="ligands">${st.ligands.slice(0,6).map(l=>`<span class="chip">${esc(l.comp_id)} ${esc(l.name||'')}</span>`).join('')}</div>`:''}
    ${st.limitations?`<div class="lim">${esc(st.limitations)}</div>`:''}
    <button class="btn view3d" data-sid="${st.id}" data-src="${esc(st.source_id)}" data-kind="${st.source}">View in 3D</button>
  </div>`).join('') : `<div class="placeholder">No structures yet — the structural biologist acquires them from PDB/AlphaFold during research cycles.</div>`;
  return `<h1>Structural Biology</h1>
    <div class="sub">Experimental PDB entries and AlphaFold models per target. Predicted models are labelled and never treated as facts.</div>
    <div class="struct-grid">${rows}</div>
    <div id="viewer-modal" class="modal" hidden><div class="modal-box">
      <div class="modal-head"><span id="viewer-title"></span><button class="btn" id="viewer-close">Close</button></div>
      <div id="viewer-3d" class="viewer-3d"></div>
      <div class="modal-foot">Source: <span id="viewer-src"></span></div>
    </div></div>`;
}

let _3dmolLoaded = null;
function load3dmol(){ if(!_3dmolLoaded) _3dmolLoaded = new Promise((res,rej)=>{const s=document.createElement('script');s.src='/static/vendor/3dmol.min.js';s.onload=res;s.onerror=()=>rej(new Error('3dmol load failed'));document.head.appendChild(s);}); return _3dmolLoaded; }

async function openViewer(sid, srcId, kind){
  const modal = document.getElementById('viewer-modal');
  modal.hidden = false;
  document.getElementById('viewer-title').textContent = (kind==='pdb'?'PDB ':'AlphaFold ')+srcId+' \u2014 3D viewer';
  document.getElementById('viewer-src').textContent = kind==='pdb'?'Protein Data Bank (experimental coordinates)':'AlphaFold DB (predicted model \u2014 hypothesis-grade, not experimental)';
  const holder = document.getElementById('viewer-3d');
  holder.innerHTML = "<div class='muted'>Loading viewer\u2026</div>";
  try{
    await load3dmol();
    const meta = await api(`/structures/${sid}`);
    const txt = await (await fetch(`/structures/${sid}/file`)).text();
    holder.innerHTML = '';
    const viewer = $3Dmol.createViewer(holder, {backgroundColor:'#0d1117'});
    viewer.addModel(txt, "mmcif");
    viewer.setStyle({}, {cartoon:{color:'spectrum'}});
    viewer.setStyle({hetflag:true}, {stick:{radius:0.25}, sphere:{radius:1.2, opacity:0.9}});
    viewer.zoomTo();
    viewer.render();
    viewer.zoom(0.8); viewer.render();
  }catch(e){ holder.innerHTML = "<div class='muted'>viewer error: "+esc(String(e.message||e))+"</div>"; }
}

function bindStructures(){
  document.querySelectorAll('.view3d').forEach(b=> b.onclick = ()=> openViewer(b.dataset.sid, b.dataset.src, b.dataset.kind));
  const c = document.getElementById('viewer-close');
  if(c) c.onclick = ()=> document.getElementById('viewer-modal').hidden = true;
}


async function renderCandidates(){
  const R = await api(`/workspaces/${state.ws.id}/candidates`);
  const C = R.candidates || [];
  if (!C.length) return '<div class="empty">No compounds catalogued yet.</div>';
  const DIMS = ["mechanism_relevance","target_confidence","binding_evidence","structural_coverage","drug_likeness","chemical_provenance","uncertainty"];
  const bars = d => DIMS.map(k=>{
    const v = d[k]||0;
    return `<div class="dimrow"><span class="dimlbl">${k.replace(/_/g,' ')}</span><span class="dimbar"><i style="width:${Math.round(v*100)}%"></i></span><span class="dimval">${v.toFixed(2)}</span></div>`;
  }).join("");
  return `<div class="section-title">Candidate ranking — multi-factor triage (not biological proof)</div>
    <div class="meta">Composite = weighted mean of inspectable dimensions; weights via <code>?weights={...}</code>.</div>` +
    C.map(c=>`<div class="struct-card">
      <div class="struct-head"><b>${esc(c.compound_name)}</b> &rarr; ${esc(c.target||'?')}
        <span class="badge established">composite ${c.composite.toFixed(2)}</span></div>
      <div class="meta">best docking ${c.best_docking!=null?c.best_docking.toFixed(2)+' kcal/mol':'—'}</div>
      <div class="dims">${bars(c.dimensions)}</div></div>`).join("");
}

async function renderSimulations(){
  const R = await api(`/workspaces/${state.ws.id}/simulations`);
  const rows = R.length ? R.map(r=>{
    const isMd = r.kind==='md';
    const unit = isMd ? 'nm' : 'kcal/mol';
    const label = isMd ? 'stability (RMSD)' : 'score';
    return `<div class="struct-card">
    <div class="struct-head"><span class="badge ${isMd?'speculative':'established'}">${esc(r.kind)}</span>
      <b>${esc(r.compound_name||'protein')}</b> &middot; ${esc(r.software)} ${esc(r.software_version)}</div>
    <div class="meta">${label} ${r.score!=null?r.score.toFixed(3)+' '+unit:'—'} &middot; seed ${r.seed} &middot; ${r.runtime_s?r.runtime_s.toFixed(1)+'s':''} &middot; ${esc(r.hardware||'')} &middot; status ${esc(r.status)}</div>
    ${r.error?`<div class="lim">${esc(r.error)}</div>`:''}
    ${r.assumptions?`<div class="lim">${esc(r.assumptions)}</div>`:''}
    <button class="btn viewpose" data-sid="${r.id}" ${isMd||!r.pose_path?'disabled':''}>${isMd?'trajectory (npz)':'View pose in 3D'}</button>
  </div>`}).join('') : `<div class="placeholder">No simulation runs yet — the research director schedules docking against structurally-covered targets during research cycles.</div>`;
  return `<h1>Simulations</h1>
    <div class="sub">Computational predictions — docking scores are NOT proof of biological activity. Each run records full reproducibility (software, version, seed, input hashes, pose artifact, runtime, hardware).</div>
    <div class="struct-grid">${rows}</div>
    <div id="pose-modal" class="modal" hidden><div class="modal-box">
      <div class="modal-head"><span id="pose-title"></span><button class="btn" id="pose-close">Close</button></div>
      <div id="pose-3d" class="viewer-3d"></div>
      <div class="modal-foot">Predicted pose (PDBQT) &mdash; docking prediction only.</div>
    </div></div>`;
}
async function openPoseViewer(sid, label){
  const modal = document.getElementById('pose-modal');
  modal.hidden = false;
  document.getElementById('pose-title').textContent = label+' — predicted pose';
  const holder = document.getElementById('pose-3d');
  holder.innerHTML = "<div class='muted'>Loading viewer…</div>";
  try{
    await load3dmol();
    const meta = await api(`/simulations/${sid}`);
    const txt = await (await fetch(`/simulations/${sid}/pose`)).text();
    holder.innerHTML = '';
    const viewer = $3Dmol.createViewer(holder, {backgroundColor:'#0d1117'});
    viewer.addModel(txt, "pdbqt");
    viewer.setStyle({}, {stick:{radius:0.22}, sphere:{radius:0.65}});
    viewer.setStyle({hetflag:false}, {sphere:{colorscheme:'Jmol'}});
    viewer.zoomTo(); viewer.render(); viewer.zoom(1.0); viewer.render();
  }catch(e){ holder.innerHTML = "<div class='muted'>pose viewer error: "+esc(String(e.message||e))+"</div>"; }
}
function bindSimulations(){
  document.querySelectorAll('.viewpose').forEach(b=> { if(!b.disabled) b.onclick = ()=> openPoseViewer(b.dataset.sid, b.textContent.trim()); });
  const c = document.getElementById('pose-close');
  if(c) c.onclick = ()=> document.getElementById('pose-modal').hidden = true;
}

async function renderLiterature(){
  if (!state.ws) return "<div class='placeholder'>No workspace.</div>";
  const ps = await api(`/workspaces/${state.ws.id}/papers`);
  if (!ps.length) return "<div class='placeholder'>No papers yet.</div>";
  const rows = ps.map(p=>`<tr data-pid="${p.id}" class="paper-row" style="cursor:pointer">
    <td>${esc(p.title.slice(0,90))}</td><td>${esc(p.journal.slice(0,24))}</td><td>${p.year||"–"}</td>
    <td><span class="badge ${badgeCls(p.status)}">${esc(p.status)}</span></td><td>${p.n_claims}</td></tr>`).join("");
  return `<h1>Literature / Evidence</h1><div class="sub">Click a paper to inspect extracted claims and provenance</div>
  <table><thead><tr><th>Title</th><th>Source</th><th>Year</th><th>Status</th><th>Claims</th></tr></thead><tbody>${rows}</tbody></table>
  <div id="paper-detail" style="margin-top:16px"></div>`;
}
function badgeCls(s){ return s==="extracted"?"established":(s==="failed"?"computational":"speculative"); }
function bindLiterature(){
  $$(".paper-row").forEach(tr=> tr.onclick = async ()=>{
    const d = await api("/papers/"+tr.dataset.pid);
    const stmts = (d.statements||[]).map(s=>`<div class="card" style="margin:6px 0">
      <div class="small">${esc(s.relation||"")} ${d.claim_type||""} <span class="badge ${badgeCls(d.status)}">${esc(s.evidence_level)}</span>
        <span class="mono muted small">${esc(s.species)} / ${esc(s.model_context)} · conf ${s.confidence}${s.contradicted?" · CONTRADICTED":""}</span></div>
      <div style="margin-top:4px">${esc(s.text)}</div></div>`).join("") || "<div class='placeholder'>no claims extracted</div>";
    $("#paper-detail").innerHTML = `<div class="card">
      <h4>${esc(d.title)}</h4>
      <div class="el"><span class="el-k">Journal</span><span>${esc(d.journal)} (${d.year||"?"})</span></div>
      <div class="el"><span class="el-k">PMID</span><span class="mono">${esc(d.pmid)}</span></div>
      <div class="el"><span class="el-k">DOI</span><span class="mono">${esc(d.doi)}</span></div>
      <div class="el"><span class="el-k">URL</span><span><a href="${esc(d.url)}" target="_blank">${esc(d.url)}</a></span></div>
      <h3 style="margin-top:12px">Extracted claims</h3>${stmts}</div>`;
  });
}

async function renderHypotheses(){
  if (!state.ws) return "<div class='placeholder'>No workspace.</div>";
  const hs = await api(`/workspaces/${state.ws.id}/hypotheses`);
  if (!hs.length) return "<div class='placeholder'>No hypotheses yet.</div>";
  const cards = hs.map(h=>{ const p=h.evidence_profile||{};
    return `<div class="card" style="grid-column:1/-1">
      <div style="display:flex;justify-content:space-between;gap:8px"><h4>${esc(h.title)}</h4>
        <span class="pill">${esc(h.status)}</span></div>
      <div class="small muted">confidence ${h.confidence} · ${h.n_support} supporting / ${h.n_counter} counter statements</div>
      ${confBar(p.overall_confidence)}
      <p>${esc(h.body)}</p>
      <div class="small"><b>Mechanism:</b> ${esc(h.mechanism_reasoning)}</div>
      <div class="small"><b>Predicted effect:</b> ${esc(h.predicted_effect)}</div>
      <div class="small"><b>Uncertainty:</b> ${esc(h.uncertainty)}</div>
      ${(h.testable_predictions||[]).map(x=>`<div class="small" style="margin-left:10px">• ${esc(x)}</div>`).join("")}
      <div class="small" style="margin-top:6px"><b>Next experiment:</b> <span class="mono">${esc(h.next_experiment)}</span></div>
      <div class="small muted" style="margin-top:6px">evidence: sources ${p.independent_sources??"–"} · contradiction ratio ${p.contradiction_ratio??"–"} · model relevance ${p.model_relevance??"–"}</div>
    </div>`; }).join("");
  return `<h1>Research Hypotheses</h1><div class="sub">Every hypothesis is grounded only in collected evidence and records its uncertainty + next experiment</div><div class="cards">${cards}</div>`;
}

async function renderQueue(){
  if (!state.ws) return "<div class='placeholder'>No workspace.</div>";
  const q = await api(`/workspaces/${state.ws.id}/queue`);
  if (!q.length) return "<div class='placeholder'>Experiment queue empty.</div>";
  const rows = q.map(e=>`<tr>
    <td><span class="badge ${badgeCls(e.status)}">${esc(e.status)}</span></td>
    <td><b>${esc(e.kind)}</b><br><span class="small muted">${esc(e.question.slice(0,70))}</span></td>
    <td>${e.priority}</td><td>${e.expected_information_gain}/${e.scientific_importance}/${e.confidence_impact}/${e.computational_cost}</td>
    <td>${esc(e.worker)}</td>
    <td class="small muted">${esc((e.result_summary||"").slice(0,40))}</td></tr>`).join("");
  return `<h1>Research Queue</h1><div class="sub">Prioritized experiments: Priority = EIG × importance × confidence-impact ÷ cost</div>
  <table><thead><tr><th>Status</th><th>Experiment</th><th>Priority</th><th>EIG/Imp/Conf/Cost</th><th>Agent</th><th>Result</th></tr></thead><tbody>${rows}</tbody></table>`;
}

async function renderTimeline(){
  if (!state.ws) return "<div class='placeholder'>No workspace.</div>";
  const tl = await api(`/workspaces/${state.ws.id}/timeline?limit=120`);
  if (!tl.length) return "<div class='placeholder'>No research activity yet.</div>";
  const items = tl.map(t=>`<div class="card" style="margin:6px 0;padding:10px 14px">
     <div class="small mono muted">${esc(t.ts)} · ${esc(t.agent)}</div>
     <div>${esc(t.message)}</div></div>`).join("");
  return `<h1>Research Timeline</h1><div class="sub">Chronological log of autonomous behaviour</div><div id="timeline-list">${items}</div>`;
}

async function renderSystem(){
  const s = await api("/status");
  const llm = s.llm||{};
  return `<h1>System Status</h1>
  <div class="cards">
    <div class="card"><div class="k">Service</div><div class="big">${esc(s.service)}</div><div class="small muted">v${esc(s.version)}</div></div>
    <div class="card"><div class="k">LLM endpoint</div><div class="big">${esc(llm.model||"—")}</div><div class="small mono muted">${esc(llm.base_url||"—")}</div></div>
    <div class="card"><div class="k">Vision</div><div class="big">${llm.support_vision?"✓ yes":"—"}</div></div>
    <div class="card"><div class="k">Context window</div><div class="big">${esc(llm.context_window||"—")}</div></div>
    <div class="card"><div class="k">Embeddings</div><div class="big">${s.embeddings?.enabled?"on":"off"}</div></div>
    <div class="card"><div class="k">Workspaces</div><div class="big">${s.workspaces}</div></div>
    <div class="card"><div class="k">Active research loops</div><div class="big">${s.active_research_loops}</div></div>
  </div>
  <div class="card" style="margin-top:12px"><div class="k">Storage</div><div class="mono small" style="margin-top:6px">${esc(s.storage?.db||"—")}</div></div>`;
}

/* ---------------- activity panel ---------------- */
async function refreshActivity(){
  if (!state.ws) return;
  try {
    const a = await api(`/workspaces/${state.ws.id}/activity`);
    const acts = a.active_experiments||[];
    let html = (acts.length? acts.map(x=>`<div class="evt"><div class="m"><b>${esc(x.worker)}</b> · ${esc(x.kind)}</div><div class="small muted">${esc(x.question.slice(0,90))}</div></div>`).join("") : "");
    const la = a.last_activity;
    if (la) html += `<div class="evt"><div class="t">${esc(la.ts)}</div><div class="m"><b>${esc(la.agent)}</b> — ${esc(la.message)}</div></div>`;
    $("#activity").innerHTML = html || "<div class='placeholder'>idle — no active research</div>";
  } catch(e){ /* ignore */ }
}

/* ---------------- boot ---------------- */
async function refresh(){
  await refreshResearchState();
  await render();
  refreshActivity();
}

(async function boot(){
  try {
    const s = await api("/status");
    $("#llm-label").textContent = (s.llm&&s.llm.model)||"llm …";
    $("#llm-dot").className = "dot " + ((s.llm&&(s.llm.error===undefined))?"ok":"bad");
  } catch(e){ $("#llm-dot").className="dot bad"; $("#llm-label").textContent="offline"; }
  $("#nav").querySelectorAll("a").forEach(a=> a.onclick = async ()=>{
    $$("#nav a").forEach(x=>x.classList.remove("active")); a.classList.add("active");
    state.view = a.dataset.view; await render();
  });
  bindResearch();
  await refreshWorkspaces();
  setInterval(refreshActivity, 5000);
})();

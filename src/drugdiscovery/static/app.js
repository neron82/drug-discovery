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

function bindView(view){
  if (view==="map") initGraph();
  if (view==="compounds") bindCompound();
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
  });
}

async function renderTargets(){
  if (!state.ws) return "<div class='placeholder'>No workspace.</div>";
  const ts = await api(`/workspaces/${state.ws.id}/targets`);
  if (!ts.length) return "<div class='placeholder'>No evidence-supported targets yet.</div>";
  const rows = ts.map(t=>`<tr>
    <td><b>${esc(t.name)}</b> <span class="badge">${esc(t.kind)}</span><br><span class="small muted">${esc((t.ligands||[]).join(", ") || "no ligands mapped")}</span></td>
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
      <button id="btn-sim" class="btn btn-primary">Search</button>
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

async function renderStructures(){ return `<h1>Structural Biology</h1><div class="sub">Phase 3 module · PDB/AlphaFold structures, binding pockets, 3D viewer (Mol*/3Dmol). Not yet implemented.</div><div class="placeholder">Scheduled for Phase 3</div>`; }
async function renderSimulations(){ return `<h1>Simulations</h1><div class="sub">Phase 4 module · docking (Vina/smina), MD (OpenMM). Queued simulation-type experiments appear here.</div><div class="placeholder">Scheduled for Phase 4</div>`; }

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

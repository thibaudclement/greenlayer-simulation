import argparse, math, random, json, os, csv
from collections import deque
import heapq
from pathlib import Path

# =============================================================================
# simulate.py
# -----------------------------------------------------------------------------
# Purpose
#   Run SDN energy simulations over a weekly traffic envelope to evaluate:
#     - Node sleeping strategies (static/adaptive/none) with guardrails
#     - Hardware-upgrade scenarios (power/perf scaling near upgraded nodes)
#   The simulator emits CSVs and JSON summaries under ./out/ for analysis.
#
# Main capabilities
#   - Weekly profile blending residential & business traffic ("res", "biz")
#   - Optional events that multiply hourly load over specified ranges
#   - Sleep policies (static/adaptive) with:
#       • Consolidation (keep a "spine" ON, sleep low-use candidates)
#       • Hysteresis to prevent rapid flapping
#       • Fixed or adaptive throughput guards (SLO-aware)
#       • Link-level EEE with hysteresis and utilization thresholds
#   - Hardware sweep: vary % of upgraded nodes; apply power/perf multipliers
#   - Timeseries emission (per tick) when requested
#
# Outputs (all under ./out/)
#   - chart1_sleep_energy.csv        : energy vs sleep %
#   - chart2_sleep_latency.csv       : latency (avg/p50/p95) vs sleep %
#   - chart3_sleep_drops.csv         : total drops vs sleep %
#   - chart4_upgrade_energy.csv      : energy vs upgraded %
#   - chart5_upgrade_latency.csv     : latency (avg/p50/p95) vs upgraded %
#   - chart6_upgrade_drops.csv       : total drops vs upgraded %
#   - ops_sleep_{pct}.json           : per-sleep% operational metrics
#   - timeseries_sleep_{pct}.csv     : optional per-tick series (if enabled)
#   - run_summary.json               : manifest with CLI args & run notes
#
# Inputs
#   - Weekly profile CSV (default: in/traffic_profile_us.csv) with columns:
#       time_bin, weekday, hour, res, biz
#     If missing, a deterministic fallback weekly envelope is used.
#   - Optional events CSV:
#       start_hour, duration_hours, multiplier
#   - Optional traffic trace CSV (advanced; replaces the weekly envelope)
#
# Reproducibility
#   - Deterministic given the --seed and static inputs (no external randomness).
#   - The baseline ranking pass and per-tick routing use the seeded RNG.
# =============================================================================


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def log(msg: str): 
    """Print with flush for progress updates in long runs."""
    print(msg, flush=True)

def seeded(seed=42): 
    """Set global RNG seed for reproducibility."""
    random.seed(seed)

def ensure_dir(p): 
    """Create directory tree if it does not exist (no-op if exists)."""
    os.makedirs(p, exist_ok=True)

def percentile(values, q):
    """
    Compute the q-th percentile with linear interpolation.
    Returns +inf if the list is empty (safe sentinel for ops metrics).
    """
    if not values: return float('inf')
    xs = sorted(values); k = (len(xs)-1)*(q/100.0)
    f=int(math.floor(k)); c=int(math.ceil(k))
    if f==c: return xs[f]
    return xs[f]*(c-k)+xs[c]*(k-f)


# -----------------------------------------------------------------------------
# Weekly traffic profile
# -----------------------------------------------------------------------------
class WeekProfile:
    """
    Diurnal/weekly traffic envelope combining residential and business series.
    - If a CSV is provided (path exists), it reads normalized columns 'res' and 'biz'
      for each hour bin (expected period = 168). 
    - Otherwise, falls back to a deterministic synthetic 7×24 profile.
    - Optional 'events' apply multiplicative factors over hour ranges.

    Parameters
    ----------
    path : str or None
        Path to a CSV with 'res' & 'biz' columns; if None or missing, use fallback.
    res_share : float
        Weight of residential load when blending res/biz into a single factor.
    events : list[(start_hour, duration_hours, multiplier)]
        Hour-indexed multiplicative events over the 0..167 weekly cycle.
    """
    def __init__(self, path=None, res_share=0.8, events=None):
        self.res_share = res_share
        self.events = events or []
        if path and os.path.exists(path):
            # Load external weekly profile
            self.rows = []
            with open(path) as f:
                r = csv.DictReader(f)
                for row in r:
                    self.rows.append({"res": float(row["res"]), "biz": float(row["biz"])})
            self.period = len(self.rows)  # typically 168
        else:
            # Deterministic fallback: shape res/biz with plausible patterns
            self.rows = []
            for d in range(7):
                for h in range(24):
                    # Residential: evening peak, late-night valley
                    res = 0.55 + 0.45*max(0, math.sin((h-6)/18*math.pi))
                    if 2 <= h <= 5: res *= 0.75
                    # Business: daytime Gaussian bump around lunch, lower weekends
                    biz = 0.1
                    if 7 <= h <= 18: biz = 0.3 + 0.7*math.exp(-((h-13)/4.0)**2)
                    if d in (5,6): biz *= 0.4
                    self.rows.append({"res": res, "biz": biz})
            self.period = len(self.rows)

    def hour_factor(self, hour_idx):
        """
        Blend res/biz for a given hour-of-week and apply any active events.
        Returns a normalized multiplier in [0, ~1+] (after events).
        """
        base = self.res_share*self.rows[hour_idx%self.period]["res"] + \
               (1.0-self.res_share)*self.rows[hour_idx%self.period]["biz"]
        mult = 1.0
        for (start, dur, m) in self.events:
            if start <= (hour_idx % self.period) < start+dur:
                mult *= m
        return base * mult


# -----------------------------------------------------------------------------
# Events loader
# -----------------------------------------------------------------------------
def load_events(path):
    """
    Load optional event multipliers:
      CSV columns: start_hour, duration_hours, multiplier
    Empty list if no file.
    """
    if not path or not os.path.exists(path): return []
    ev=[]
    with open(path) as f:
        r=csv.DictReader(f)
        for row in r:
            ev.append((int(row["start_hour"]), int(row["duration_hours"]), float(row["multiplier"])))
    return ev


# -----------------------------------------------------------------------------
# Trace driver
# -----------------------------------------------------------------------------
class TraceDriver:
    """
    Drives per-tick demands from a (time_bin, src, dst, gbits) CSV trace.
    - Lazily builds a label->node_id map (or uses provided node_map CSV).
    - Aggregates demands per tick; supports wrap-around reuse.

    Parameters
    ----------
    path : str
        CSV path with columns: time_bin, src, dst, gbits
    node_map_path : str or None
        Optional CSV mapping: label,node_id
    n_nodes : int
        Number of nodes in the graph (for label assignment bounds).
    scale : float
        Multiplicative factor applied to 'gbits' from the trace.
    wrap : bool
        Whether to wrap around after the last time_bin during simulation.
    """
    def __init__(self, path, node_map_path, n_nodes, scale=1.0, wrap=True):
        import csv as _csv
        self.scale = scale; self.wrap = wrap; self.n_nodes = n_nodes
        self.label_to_id = {}

        # Optionally load explicit label->id mapping
        if node_map_path:
            with open(node_map_path) as f:
                r = _csv.DictReader(f)
                for row in r:
                    self.label_to_id[row["label"]] = int(row["node_id"])

        # Load traffic by time_bin, constructing implicit mapping if needed
        tmp = {}; max_bin=-1
        with open(path) as f:
            r=_csv.DictReader(f)
            for row in r:
                t=int(row["time_bin"]); s=row["src"]; d=row["dst"]; gb=float(row["gbits"])*scale
                max_bin=max(max_bin,t); tmp.setdefault(t,[]).append((s,d,gb))

        # Normalize to dense time bins 0..max_bin and map labels to node ids
        self.bins=[]
        for t in range(max_bin+1):
            lst=[]
            for (s,d,gb) in tmp.get(t,[]):
                if s not in self.label_to_id:
                    if len(self.label_to_id) >= n_nodes: raise ValueError(f"Need mapping for '{s}'")
                    self.label_to_id[s]=len(self.label_to_id)
                if d not in self.label_to_id:
                    if len(self.label_to_id) >= n_nodes: raise ValueError(f"Need mapping for '{d}'")
                    self.label_to_id[d]=len(self.label_to_id)
                lst.append((self.label_to_id[s], self.label_to_id[d], gb))
            self.bins.append(lst)
        self.period=len(self.bins)

    def demands_for_tick(self,t):
        """Return list[(src_id, dst_id, gbits)] for tick t, wrapping if enabled."""
        if self.period==0: return []
        if t<self.period: return self.bins[t]
        return self.bins[t%self.period]


# -----------------------------------------------------------------------------
# Graph / topology
# -----------------------------------------------------------------------------
class Graph:
    """Undirected graph with per-edge latency/capacity; adjacency lists per node."""
    def __init__(self, n: int):
        self.n=n; self.adj=[[] for _ in range(n)]
        self.edges=[]; self._emap={}
    def _add_edge_undirected(self,u,v,lat,cap):
        key=frozenset((u,v))
        if key in self._emap: return self._emap[key]
        eid=len(self.edges); self._emap[key]=eid
        self.edges.append({"u":u,"v":v,"lat_ms":lat,"cap_gbps":cap,"base_lat_ms":lat,"base_cap_gbps":cap})
        return eid
    def add_edge(self,u,v,lat,cap):
        """Add undirected edge once; maintain adjacency for both endpoints."""
        eid=self._add_edge_undirected(u,v,lat,cap)
        self.adj[u].append((v,eid)); self.adj[v].append((u,eid))

def make_scale_free(n, m0=3, m=2, lat_range=(0.2,5.0), caps=(1,2,10,40)):
    """
    Barabási–Albert-ish scale-free generator:
      - Start with m0-clique, then preferentially attach m edges per new node.
      - Randomize per-edge latency and capacity from provided ranges.
    """
    g=Graph(n); deg=[0]*n
    # Seed clique
    for i in range(m0):
        for j in range(i+1,m0):
            g.add_edge(i,j,random.uniform(*lat_range),random.choice(caps))
            deg[i]+=1; deg[j]+=1
    total=sum(deg)
    # Preferential attachment
    for v in range(m0,n):
        targets=set()
        while len(targets)<m:
            r=random.uniform(0,total or 1); s=0; pick=None
            for u,d in enumerate(deg):
                s+= (d or 0.0001)
                if s>=r: pick=u; break
            if pick is None: pick=random.randrange(v)
            targets.add(pick)
        for u in targets:
            g.add_edge(v,u,random.uniform(*lat_range),random.choice(caps))
            deg[v]+=1; deg[u]+=1
        total=sum(deg)
    return g

def make_fat_tree(n, lat_range=(0.1,1.0), caps=(10,40)):
    """
    Coarse fat-tree-ish fabric:
      - Partition nodes into core / aggregation / edge tiers.
      - Randomized inter-tier connectivity to avoid degenerate graphs.
    """
    g=Graph(n); core=max(2,n//4); agg=max(2,n//4); edge=n-core-agg
    core_ids=list(range(core)); agg_ids=list(range(core,core+agg)); edge_ids=list(range(core+agg,n))
    for c in core_ids:
        for a in random.sample(agg_ids,k=max(2,len(agg_ids)//2)):
            g.add_edge(c,a,random.uniform(*lat_range),random.choice(caps))
    for a in agg_ids:
        for e in random.sample(edge_ids,k=max(2,len(edge_ids)//4)):
            g.add_edge(a,e,random.uniform(*lat_range),random.choice(caps))
    for _ in range(max(1,edge//3)):
        u,v=random.sample(edge_ids,2)
        g.add_edge(u,v,random.uniform(*lat_range),random.choice(caps))
    return g


# -----------------------------------------------------------------------------
# Articulation points
# -----------------------------------------------------------------------------
def articulation_points(g: Graph):
    """Classic DFS-based articulation point detection (Tarjan)."""
    n=g.n; timer=0; tin=[-1]*n; low=[-1]*n; vis=[False]*n; aps=set()
    def dfs(v,p=-1):
        nonlocal timer
        vis[v]=True; tin[v]=low[v]=timer; timer+=1; ch=0
        for (to,_) in g.adj[v]:
            if to==p: continue
            if vis[to]: low[v]=min(low[v], tin[to])
            else:
                dfs(to,v); low[v]=min(low[v], low[to]); ch+=1
                if p!=-1 and low[to]>=tin[v]: aps.add(v)
        if p==-1 and ch>1: aps.add(v)
    for i in range(n):
        if not vis[i]: dfs(i)
    return aps


# -----------------------------------------------------------------------------
# Shortest path (Dijkstra by latency)
# -----------------------------------------------------------------------------
def shortest_path(g: Graph, src, dst, alive_nodes, alive_links):
    """
    Dijkstra over alive nodes/links by latency weight.
    Returns (nodes, edges) sequence or None if disconnected.
    """
    if src==dst: return []
    INF=1e18; n=g.n
    dist=[INF]*n; prev=[(-1,-1)]*n
    dist[src]=0.0; pq=[(0.0,src)]
    while pq:
        d,u=heapq.heappop(pq)
        if d!=dist[u]: continue
        if u==dst: break
        if not alive_nodes[u]: continue
        for v,eid in g.adj[u]:
            if not alive_nodes[v] or not alive_links[eid]: continue
            nd=d + g.edges[eid]["lat_ms"]
            if nd<dist[v]:
                dist[v]=nd; prev[v]=(u,eid); heapq.heappush(pq,(nd,v))
    if dist[dst]==INF: return None
    nodes=[]; edges=[]; u=dst
    while u!=-1 and u!=src:
        p,eid=prev[u]
        if p==-1: return None
        nodes.append(u); edges.append(eid); u=p
    nodes.append(src); nodes.reverse(); edges.reverse()
    return nodes, edges


# -----------------------------------------------------------------------------
# Node / Link state & power models
# -----------------------------------------------------------------------------
class Node:
    """Switch/router node with simple power model and sleep/wake state."""
    def __init__(self, idx, is_efficient=False, params=None):
        self.idx=idx; self.on=True; self.sleeping=False; self.wake_until=0
        self.is_efficient=is_efficient; self.params=params or {}; self.energy_Wh=0.0
        # ops: track sleep durations to detect flaps
        self._sleep_start = None
    def power_W(self, load_frac):
        """
        Piecewise-linear(ish) model:
          P = P_idle + P_dyn * load_frac
          - Efficient nodes apply multiplicative reductions.
          - Sleeping nodes draw P_sleep; off nodes draw 0.
        """
        p_idle=self.params['P_idle_std']; p_dyn=self.params['P_dyn_std']
        if self.is_efficient:
            p_idle*=self.params['efficient_idle_factor']; p_dyn*=self.params['efficient_dyn_factor']
        if not self.on: return 0.0
        if self.sleeping: return self.params['P_sleep']
        return p_idle + p_dyn*max(0.0,min(1.0,load_frac))

class LinkState:
    """Per-link EEE-style sleep with simple idle/sleep power model."""
    def __init__(self, eid, params):
        self.eid=eid; self.sleeping=False; self.wake_until=0
        self.params=params; self.energy_Wh=0.0; self.util_gbps=0.0
    def power_W(self): 
        return self.params['P_link_sleep'] if self.sleeping else self.params['P_link_idle']


# -----------------------------------------------------------------------------
# Simulator core
# -----------------------------------------------------------------------------
class Simulator:
    """
    Core simulation engine:
      - Routes demands each tick over currently-alive nodes/links.
      - Decides node/link sleep states under policy + guardrails.
      - Tracks energy, drops, latency stats, and operational metrics.
    """
    def __init__(self, g: Graph, params, policy, sleep_cap_percent=0,
                 hysteresis=5, forbidden_sleep_nodes=None, seed=42,
                 slo_window=30, max_drop_rate=0.02, max_latency_ms=50.0,
                 throughput_margin=0.8, avoid_articulation=True,
                 link_sleep_rho=0.02, link_hysteresis=5,
                 trace_driver=None, week_profile=None, days=1, tick_min=60,
                 heavy_user_share=0.0, emit_timeseries=False):
        seeded(seed)
        self.g=g; self.params=params
        self.nodes=[Node(i, False, params) for i in range(g.n)]
        self.links=[LinkState(eid, params) for eid in range(len(self.g.edges))]
        self.policy=policy; self.sleep_cap=int(g.n*sleep_cap_percent/100.0)
        self.hysteresis=hysteresis; self.link_hysteresis=link_hysteresis
        self.last_state_change=[-9999]*g.n; self.last_link_change=[-9999]*len(self.links)
        self.node_load=[0.0]*self.g.n
        self.forbidden_sleep_nodes=set(forbidden_sleep_nodes) if (avoid_articulation and forbidden_sleep_nodes) else set()
        # SLO (rolling)
        self.slo_window=slo_window; self.max_drop_rate=max_drop_rate; self.max_latency_ms=max_latency_ms
        self.roll_drops=deque(maxlen=slo_window); self.roll_sent=deque(maxlen=slo_window)
        self.roll_latency_sum=deque(maxlen=slo_window); self.roll_latency_cnt=deque(maxlen=slo_window)
        # Guards
        self.throughput_margin=throughput_margin
        self.link_sleep_rho=link_sleep_rho
        self.baseline_edge_cap=sum(e["cap_gbps"] for e in self.g.edges)
        # Traffic source(s)
        self.trace_driver=trace_driver; self.week_profile=week_profile
        self.days=days; self.tick_min=tick_min
        self.heavy_user_share=heavy_user_share
        # Reporting accumulators
        self._sleep_nodes_sum=0.0; self._alive_cap_sum=0.0; self._ticks_count=0; self._latency_samples=[]
        # Optional timeseries emission
        self.emit_timeseries = emit_timeseries
        self._ts_rows = [] if emit_timeseries else None
        # Adaptive guard state
        self._last_total_link_util = None
        self.guard_mode = "fixed"  # overridden by runner
        self.params_guard_rho_target = 0.60
        self.params_guard_headroom  = 1.20
        self.spine_frac = 0.20
        # Ops metrics (we collect per-tick maxima to summarize with p95)
        self._ops_max_rho_series = []
        self._ops_max_q_series   = []
        self._ops_node_sleeps = 0
        self._ops_node_wakes  = 0
        self._ops_node_flaps  = 0
        self._ops_slo_breach_ticks = 0

    # traffic per tick
    def global_load(self, tick, total_ticks):
        """
        Hourly envelope:
          - Prefer WeekProfile (hour-of-week blend).
          - Else fallback to single-day sinusoid (legacy).
        """
        if self.week_profile:
            ticks_per_hour = max(1, int(60/self.tick_min))
            hour_idx = (tick // ticks_per_hour) % (7*24*self.days if self.days>1 else 7*24)
            return self.week_profile.hour_factor(hour_idx % (7*24))
        # fallback single-day envelope
        t=(tick/total_ticks)*24.0
        minl=self.params.get('min_load',0.3); peak=self.params.get('peak_hour',18.0)
        return max(0.0,min(1.0, minl + (1.0-minl)*(math.sin(math.pi*(t-peak)/24.0))**2))

    def low_window(self, tick, total_ticks, thresh=0.5):
        """Boolean: is current load below threshold (proxy for off-peak windows)."""
        return self.global_load(tick,total_ticks) < thresh

    def baseline_utilization_profile(self, total_ticks, pairs_per_tick=200):
        """
        Run a short baseline pass with everything on to rank nodes by usage.
        Returns (ranked_nodes_by_increasing_use, use_vector).
        """
        alive=[True]*self.g.n; alive_links=[True]*len(self.g.edges); use=[0.0]*self.g.n
        for t in range(total_ticks):
            demands=self.sample_demands(pairs_per_tick, self.global_load(t,total_ticks))
            for (s,d,gb) in demands:
                sp=shortest_path(self.g,s,d,alive,alive_links)
                if not sp: continue
                nodes,_=sp
                for v in nodes: use[v]+=gb
        ranked=sorted(range(self.g.n), key=lambda v: use[v])
        return ranked, use

    def sample_demands(self, pairs, scale):
        """
        Synthetic traffic generator:
          - Random source/dest pairs with Paretian flow sizes (heavy-tail).
          - Optional 'heavy_user_share' boosts scale for a fraction of draws.
        """
        shp=self.params.get('burst_shape',2.0); scl=self.params.get('burst_scale',0.02)
        res=[]
        for _ in range(pairs):
            s=random.randrange(self.g.n); d=random.randrange(self.g.n)
            while d==s: d=random.randrange(self.g.n)
            tail = random.paretovariate(shp)
            if self.heavy_user_share>0 and random.random() < self.heavy_user_share:
                tail *= 3.0
            gb = tail*scl*scale
            res.append((s,d,gb))
        return res

    def apply_link_perf_scaling(self, efficient_nodes):
        """
        Apply capacity/latency multipliers to edges adjacent to upgraded nodes.
        Multipliers are applied once per run and reset from base_* each time.
        """
        cap_scale=self.params.get("cap_scale_eff",1.0); lat_scale=self.params.get("lat_scale_eff",1.0)
        for e in self.g.edges:
            e["cap_gbps"]=e["base_cap_gbps"]; e["lat_ms"]=e["base_lat_ms"]
        if cap_scale==1.0 and lat_scale==1.0: return
        eff=set(efficient_nodes)
        for e in self.g.edges:
            mult=(1 if e["u"] in eff else 0) + (1 if e["v"] in eff else 0)
            if mult:
                e["cap_gbps"]=e["base_cap_gbps"]*(cap_scale**mult)
                e["lat_ms"]=e["base_lat_ms"]*(lat_scale**mult)

    def alive_edge_capacity_if(self, sleep_set):
        """Sum capacity of edges whose endpoints are both awake (given sleep_set)."""
        asleep=set(sleep_set); cap=0.0
        for e in self.g.edges:
            if (e["u"] in asleep) or (e["v"] in asleep): continue
            cap += e["cap_gbps"]
        return cap

    # Adaptive guard: required capacity from last tick’s aggregated link util
    def required_capacity_now(self):
        """
        Adaptive throughput guard:
          - If no history, fall back to conservative fixed fraction of baseline.
          - Else require enough alive capacity to keep rho near target with headroom.
        """
        if self._last_total_link_util is None:
            return self.throughput_margin * self.baseline_edge_cap  # conservative first tick
        return self.params_guard_headroom * (self._last_total_link_util / max(1e-9, self.params_guard_rho_target))

    # Consolidation: keep a "spine", sleep the rest under guard constraints
    def pick_sleep_set_consolidated(self, candidates, cap_limit):
        """
        Greedy consolidation:
          - Pin a 'spine' of high-degree nodes ON.
          - Try to add low-use candidates to sleep_set as long as capacity guard holds.
        """
        deg=[len(self.g.adj[i]) for i in range(self.g.n)]
        eligible=[v for v in range(self.g.n) if v not in self.forbidden_sleep_nodes]
        keep_k = max(6, int(self.g.n * self.spine_frac))
        spine = set(sorted(eligible, key=lambda v: (-deg[v]))[:keep_k])
        sleep_set=set()
        for v in candidates:
            if v in self.forbidden_sleep_nodes or v in spine: continue
            trial=set(sleep_set); trial.add(v)
            if self.alive_edge_capacity_if(trial) >= cap_limit:
                sleep_set = trial
        return sleep_set

    def _slo_guardrail(self, low, to_sleep):
        """
        Safety override during low windows:
          If rolling drop rate or latency exceeds SLO, keep (wake) the highest-degree
          node from the proposed sleep set to restore headroom.
        """
        if not low or not to_sleep: return to_sleep
        sent=sum(self.roll_sent) or 1e-9; drops=sum(self.roll_drops)
        drop_rate=drops/sent
        lat_cnt=sum(self.roll_latency_cnt)
        lat_avg=(sum(self.roll_latency_sum)/lat_cnt) if lat_cnt>0 else 0.0
        breach = (drop_rate > self.max_drop_rate) or (lat_avg > self.max_latency_ms)
        if breach: self._ops_slo_breach_ticks += 1
        if not breach: return to_sleep
        # wake highest-degree node
        deg=lambda v: len(self.g.adj[v])
        w=max(to_sleep, key=deg)
        ts=set(to_sleep); ts.discard(w); return ts

    def run_once(self, total_ticks=1440, pairs_per_tick=200, sleep_strategy='adaptive',
                 log_progress=False, use_throughput_guard=True, consolidated=True):
        """
        Execute one simulation with current parameters/policy configuration.
        Returns summary stats and an ops_summary dict (see below).
        """
        # --- init per-run state ---
        for n in self.nodes:
            n.on=True; n.sleeping=False; n.wake_until=0; n.energy_Wh=0.0; n._sleep_start=None
        for l in self.links: l.sleeping=False; l.wake_until=0; l.energy_Wh=0.0; l.util_gbps=0.0
        alive_nodes=[True]*self.g.n; alive_links=[True]*len(self.g.edges)
        total_drops=0; latency_accum=0.0; latency_count=0
        self.roll_drops.clear(); self.roll_sent.clear(); self.roll_latency_sum.clear(); self.roll_latency_cnt.clear()
        self._sleep_nodes_sum=0.0; self._alive_cap_sum=0.0; self._ticks_count=0; self._latency_samples.clear()
        if self.emit_timeseries and self._ts_rows is None: self._ts_rows=[]
        self._ops_max_rho_series.clear(); self._ops_max_q_series.clear()
        self._ops_node_sleeps = self._ops_node_wakes = self._ops_node_flaps = 0
        self._ops_slo_breach_ticks = 0

        # Build a baseline usage ranking for static/adaptive candidate pools
        least_used_rank,_=self.baseline_utilization_profile(max(60,total_ticks//12), pairs_per_tick)

        next_mark=0.1
        cap_limit_fixed = self.throughput_margin*self.baseline_edge_cap
        ticks_per_hour = max(1, int(60/self.tick_min))

        # --- main tick loop ---
        for tick in range(total_ticks):
            if log_progress and tick/total_ticks>=next_mark:
                log(f"[sim] {int(next_mark*100)}% ticks processed"); next_mark+=0.1

            # Sleep decisions (node-level)
            low=self.low_window(tick,total_ticks)
            to_sleep=set()
            if low and self.sleep_cap>0:
                # Candidate selection
                if sleep_strategy=='static':
                    cands = least_used_rank[:self.sleep_cap]
                elif sleep_strategy=='adaptive':
                    cands = sorted(range(self.g.n), key=lambda v:self.node_load[v])[:self.sleep_cap]
                else:
                    cands=[]
                # Guards (throughput-based)
                if use_throughput_guard:
                    cap_limit = self.required_capacity_now() if self.guard_mode=="adaptive" else cap_limit_fixed
                    to_sleep = self.pick_sleep_set_consolidated(cands, cap_limit) if consolidated \
                               else set(v for v in cands if v not in self.forbidden_sleep_nodes and
                                        self.alive_edge_capacity_if([v]) >= cap_limit)
                else:
                    to_sleep = set(v for v in cands if v not in self.forbidden_sleep_nodes)

            # SLO guardrail may override a subset of the sleep set
            to_sleep=self._slo_guardrail(low,to_sleep)

            # Node hysteresis + wake timers; ops counters (sleeps/wakes/flaps)
            for v in range(self.g.n):
                now=tick; node=self.nodes[v]
                if v in to_sleep:
                    if node.on and (not node.sleeping) and now-self.last_state_change[v]>=self.hysteresis:
                        node.sleeping=True; node._sleep_start=now
                        self.last_state_change[v]=now; self._ops_node_sleeps += 1
                else:
                    if node.sleeping and now-self.last_state_change[v]>=self.hysteresis:
                        node.sleeping=False; node.wake_until=self.params['t_wake_ticks']
                        self.last_state_change[v]=now; self._ops_node_wakes += 1
                        # flap if very short sleep (< 2*hysteresis ticks)
                        if node._sleep_start is not None and (now - node._sleep_start) < (2*self.hysteresis):
                            self._ops_node_flaps += 1
                        node._sleep_start=None

            # Alive node mask (respect wake timers)
            for v in range(self.g.n):
                n=self.nodes[v]
                if n.wake_until>0: n.wake_until-=1
                alive_nodes[v]= n.on and (not n.sleeping) and (n.wake_until==0)

            # Link sleep policy (EEE-like with hysteresis)
            for eid,link in enumerate(self.links):
                now=tick; u=self.g.edges[eid]["u"]; v=self.g.edges[eid]["v"]
                force = (not alive_nodes[u]) or (not alive_nodes[v])
                if force:
                    if (not link.sleeping) and now-self.last_link_change[eid]>=self.link_hysteresis:
                        link.sleeping=True; self.last_link_change[eid]=now
                else:
                    if low and link.util_gbps < self.link_sleep_rho*max(1e-9,self.g.edges[eid]["cap_gbps"]):
                        if (not link.sleeping) and now-self.last_link_change[eid]>=self.link_hysteresis:
                            link.sleeping=True; self.last_link_change[eid]=now
                    else:
                        if link.sleeping and now-self.last_link_change[eid]>=self.link_hysteresis:
                            link.sleeping=False; link.wake_until=self.params['t_link_wake_ticks']; self.last_link_change[eid]=now

            # Alive link mask (respect wake timers)
            for eid,link in enumerate(self.links):
                if link.wake_until>0: link.wake_until-=1
                alive_links[eid] = (not link.sleeping) and (link.wake_until==0)

            # Capture last-tick aggregate link utilization for adaptive guard
            if self.links:
                self._last_total_link_util = sum(link.util_gbps for link in self.links)
            else:
                self._last_total_link_util = 0.0

            # Reset per-tick loads for next routing pass
            self.node_load=[0.0]*self.g.n
            for link in self.links: link.util_gbps=0.0

            # Demands for this tick: trace or synthetic
            if self.trace_driver: demands = self.trace_driver.demands_for_tick(tick)
            else:                 demands = self.sample_demands(pairs_per_tick, self.global_load(tick,total_ticks))

            # Route demands, accumulate drops and latencies
            sent_this=0; drops_this=0; lat_sum_this=0.0; lat_cnt_this=0
            for (s,d,gb) in demands:
                sent_this+=1
                sp=shortest_path(self.g,s,d,alive_nodes,alive_links)
                if not sp: drops_this+=1; continue
                nodes,edges=sp
                for v in nodes: self.node_load[v]+=gb
                for eid in edges: self.links[eid].util_gbps += gb
                # Latency approximation with queueing blow-up near saturation
                path_lat=0.0; ok=True
                for eid in edges:
                    e=self.g.edges[eid]; base=e["lat_ms"]; cap=e["cap_gbps"]
                    rho=self.links[eid].util_gbps / max(1e-9,cap)
                    if rho>1.2: ok=False; break
                    q= base / max(1e-3, (1.0-min(0.99,rho)))
                    path_lat += (base+q)
                if not ok:
                    drops_this+=1
                else:
                    latency_accum += path_lat; latency_count += 1
                    lat_sum_this += path_lat; lat_cnt_this += 1
                    self._latency_samples.append(path_lat)

            # Update rolling SLO windows
            self.roll_sent.append(sent_this); self.roll_drops.append(drops_this)
            self.roll_latency_sum.append(lat_sum_this); self.roll_latency_cnt.append(lat_cnt_this)

            # Energy accounting (tick_min minutes → hours)
            tick_hours = self.tick_min/60.0
            node_W = 0.0
            for v in range(self.g.n):
                load_frac=min(1.0, self.node_load[v]/self.params['node_capacity_gbps_proxy'])
                node_W += self.nodes[v].power_W(load_frac)
                self.nodes[v].energy_Wh += self.nodes[v].power_W(load_frac) * tick_hours
            link_W = 0.0
            for link in self.links:
                link_W += link.power_W()
                link.energy_Wh += link.power_W() * tick_hours

            total_drops += drops_this

            # Aggregate reporting counters
            asleep_count = sum(1 for v in range(self.g.n) if self.nodes[v].sleeping)
            alive_cap = self.alive_edge_capacity_if([v for v in range(self.g.n) if self.nodes[v].sleeping])
            self._sleep_nodes_sum += asleep_count
            self._alive_cap_sum  += (alive_cap / max(1e-9,self.baseline_edge_cap))
            self._ticks_count    += 1

            # Per-tick ops (consider only alive links): record maxima
            link_rhos=[]; link_qs=[]
            for eid,link in enumerate(self.links):
                if not alive_links[eid]: continue
                e=self.g.edges[eid]; base=e["lat_ms"]; cap=e["cap_gbps"]
                rho = link.util_gbps / max(1e-9,cap)
                q_queue = (base / max(1e-3,(1.0-min(0.99,rho)))) - base  # queue-only ms
                link_rhos.append(rho)
                link_qs.append(q_queue)
            if link_rhos:
                self._ops_max_rho_series.append(max(link_rhos))
                self._ops_max_q_series.append(max(link_qs))

            # Optional timeseries emission buffer
            if self.emit_timeseries and self._ts_rows is not None:
                hour_of_week = (tick // ticks_per_hour) % (7*24)
                tick_Wh = (node_W + link_W) * tick_hours
                self._ts_rows.append([tick, hour_of_week, tick_Wh, asleep_count,
                                      alive_cap / max(1e-9,self.baseline_edge_cap)])

        # --- summarize run ---
        total_energy_nodes=sum(n.energy_Wh for n in self.nodes)
        total_energy_links=sum(l.energy_Wh for l in self.links)
        total_energy = total_energy_nodes + total_energy_links
        avg_latency = (latency_accum/latency_count) if latency_count>0 else float('inf')
        p50 = percentile(self._latency_samples, 50.0)
        p95 = percentile(self._latency_samples, 95.0)
        avg_sleep_nodes = self._sleep_nodes_sum / max(1,self._ticks_count)
        avg_alive_cap_frac = self._alive_cap_sum / max(1,self._ticks_count)

        # ops summary for observability / reproducibility
        ops = {
            "p95_max_link_rho": percentile(self._ops_max_rho_series, 95.0) if self._ops_max_rho_series else 0.0,
            "mean_max_link_rho": (sum(self._ops_max_rho_series)/len(self._ops_max_rho_series)) if self._ops_max_rho_series else 0.0,
            "p95_max_link_queue_ms": percentile(self._ops_max_q_series, 95.0) if self._ops_max_q_series else 0.0,
            "total_node_sleeps": self._ops_node_sleeps,
            "total_node_wakes":  self._ops_node_wakes,
            "node_flaps_short":  self._ops_node_flaps,
            "slo_breach_ticks":  self._ops_slo_breach_ticks,
            "avg_sleep_nodes":   avg_sleep_nodes,
            "avg_alive_capacity_fraction": avg_alive_cap_frac
        }

        return {
            "total_energy_Wh": total_energy,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "total_drops": total_drops,
            "avg_sleep_nodes": avg_sleep_nodes,
            "avg_alive_cap_frac": avg_alive_cap_frac,
            "ops_summary": ops
        }


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------
def write_csv(path, headers, rows):
    """
    Write a simple CSV to 'path'; ensures parent directory exists.
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path,"w", newline="") as f:
        w=csv.writer(f); w.writerow(headers); [w.writerow(r) for r in rows]

def apply_hw_profile(args):
    """
    Apply predefined hardware profiles by mutating argparse Namespace.
    - std_qfx5120: conservative baseline
    - eff_arista_7050x: lower power + mild perf gains
    """
    if args.hw_profile=="std_qfx5120":
        args.P_idle_std=210.0; args.P_dyn_std=60.0; args.t_wake_ms=max(args.t_wake_ms,20)
        args.cap_scale_eff=max(args.cap_scale_eff,1.0); args.lat_scale_eff=min(args.lat_scale_eff,1.0)
    elif args.hw_profile=="eff_arista_7050x":
        args.P_idle_std=160.0; args.P_dyn_std=40.0; args.t_wake_ms=max(args.t_wake_ms,10)
        args.cap_scale_eff=max(args.cap_scale_eff,1.15); args.lat_scale_eff=min(args.lat_scale_eff,0.95)
    return args

def run_experiments(args):
    """
    Orchestrate the two sweeps:
      (1) Sleep sweep: vary sleep_cap_percent (0..90 step 10).
      (2) Hardware upgrade sweep: vary % of nodes upgraded (0..100 step 10).
    Emit CSVs and per-sleep-cap ops JSONs to ./out/
    """
    ensure_dir("out")

    # Topology
    g = make_scale_free(100) if args.topology=="scale_free" else make_fat_tree(100)
    aps = articulation_points(g) if args.avoid_articulation else set()

    # Traffic sources: either trace or weekly envelope
    trace_driver=None
    if args.trace_csv:
        trace_driver=TraceDriver(args.trace_csv, args.node_map_csv, g.n, args.trace_scale, True)
    events = load_events(args.event_csv)
    week_prof = WeekProfile(args.traffic_profile_csv, args.res_share, events) if not trace_driver else None

    # Power/perf parameters (static during a run)
    params = {
        "P_idle_std": args.P_idle_std, "P_dyn_std": args.P_dyn_std,
        "efficient_idle_factor": args.efficient_idle_factor, "efficient_dyn_factor": args.efficient_dyn_factor,
        "P_sleep": args.P_sleep,
        "t_wake_ticks": max(1, int(round(args.t_wake_ms / (args.tick_min*60*1000/60_000)))),
        "node_capacity_gbps_proxy": args.node_capacity_gbps_proxy,
        "peak_hour": args.peak_hour, "min_load": args.min_load,
        "burst_shape": args.burst_shape, "burst_scale": args.burst_scale,
        "P_link_idle": args.P_link_idle, "P_link_sleep": args.P_link_sleep,
        "t_link_wake_ticks": max(1, int(round(args.t_link_wake_ms / (args.tick_min*60*1000/60_000)))),
        "cap_scale_eff": args.cap_scale_eff, "lat_scale_eff": args.lat_scale_eff,
    }

    ticks_total = int((args.days*24*60)/args.tick_min)

    # --- Sleep sweep ---
    rows=[]
    for pct in range(0,100,10):
        if pct==100: break
        sim=Simulator(g, params, policy=args.policy, sleep_cap_percent=pct,
                      hysteresis=args.sleep_hysteresis, forbidden_sleep_nodes=aps,
                      seed=args.seed, slo_window=args.slo_window, max_drop_rate=args.max_drop_rate,
                      max_latency_ms=args.max_latency_ms, throughput_margin=args.throughput_margin,
                      avoid_articulation=args.avoid_articulation, link_sleep_rho=args.link_sleep_rho,
                      link_hysteresis=args.link_hysteresis, trace_driver=trace_driver,
                      week_profile=week_prof, days=args.days, tick_min=args.tick_min,
                      heavy_user_share=args.heavy_user_share, emit_timeseries=args.emit_timeseries)
        # Wire guard & spine params
        sim.guard_mode = args.guard_mode
        sim.params_guard_rho_target = args.rho_target
        sim.params_guard_headroom  = args.headroom
        sim.spine_frac = args.spine_frac

        sim.apply_link_perf_scaling(efficient_nodes=[])
        res=sim.run_once(total_ticks=ticks_total, pairs_per_tick=args.pairs,
                         sleep_strategy=args.policy, log_progress=args.log_progress,
                         use_throughput_guard=True, consolidated=True)
        rows.append([pct, res["total_energy_Wh"], res["avg_latency_ms"], res["p50_latency_ms"],
                     res["p95_latency_ms"], res["total_drops"], res["avg_sleep_nodes"], res["avg_alive_cap_frac"]])

        # Optional timeseries per sleep %
        if args.emit_timeseries and sim._ts_rows is not None:
            p = Path("out") / f"timeseries_sleep_{pct}.csv"
            with p.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["tick","hour_of_week","tick_Wh","sleep_nodes","alive_capacity_fraction"])
                w.writerows(sim._ts_rows)
        # OPS summary per cap
        with (Path("out") / f"ops_sleep_{pct}.json").open("w") as f:
            json.dump(res["ops_summary"], f, indent=2)

    # Charts for sleep sweep
    write_csv("out/chart1_sleep_energy.csv",
              ["sleep_percent","total_energy_Wh","avg_latency_ms","p50_latency_ms","p95_latency_ms",
               "total_drops","avg_sleep_nodes","avg_alive_capacity_fraction"],
              rows)
    write_csv("out/chart2_sleep_latency.csv",
              ["sleep_percent","avg_latency_ms","p50_latency_ms","p95_latency_ms"],
              [[r[0], r[2], r[3], r[4]] for r in rows])
    write_csv("out/chart3_sleep_drops.csv", ["sleep_percent","total_drops"], [[r[0], r[5]] for r in rows])

    # --- Hardware upgrade sweep (no sleeping, perf/power changes only) ---
    base=Simulator(g, params, policy='none', sleep_cap_percent=0,
                   hysteresis=args.sleep_hysteresis, forbidden_sleep_nodes=aps,
                   seed=args.seed, slo_window=args.slo_window, max_drop_rate=args.max_drop_rate,
                   max_latency_ms=args.max_latency_ms, throughput_margin=args.throughput_margin,
                   avoid_articulation=args.avoid_articulation, link_sleep_rho=args.link_sleep_rho,
                   link_hysteresis=args.link_hysteresis, trace_driver=trace_driver,
                   week_profile=week_prof, days=args.days, tick_min=args.tick_min,
                   heavy_user_share=args.heavy_user_share, emit_timeseries=False)
    base.guard_mode = args.guard_mode
    base.params_guard_rho_target = args.rho_target
    base.params_guard_headroom  = args.headroom
    base.spine_frac = args.spine_frac

    ranking,_=base.baseline_utilization_profile(max(60, ticks_total//12), args.pairs)

    rows2=[]
    for pct in range(0,101,10):
        sim=Simulator(g, params, policy='none', sleep_cap_percent=0,
                      hysteresis=args.sleep_hysteresis, forbidden_sleep_nodes=aps,
                      seed=args.seed, slo_window=args.slo_window, max_drop_rate=args.max_drop_rate,
                      max_latency_ms=args.max_latency_ms, throughput_margin=args.throughput_margin,
                      avoid_articulation=args.avoid_articulation, link_sleep_rho=args.link_sleep_rho,
                      link_hysteresis=args.link_hysteresis, trace_driver=trace_driver,
                      week_profile=week_prof, days=args.days, tick_min=args.tick_min,
                      heavy_user_share=args.heavy_user_share, emit_timeseries=False)
        sim.guard_mode = args.guard_mode
        sim.params_guard_rho_target = args.rho_target
        sim.params_guard_headroom  = args.headroom
        sim.spine_frac = args.spine_frac

        # Mark top-k least-used nodes as upgraded (from baseline ranking)
        eff=set(ranking[:int(g.n*pct/100.0)])
        for v in range(g.n): sim.nodes[v].is_efficient=(v in eff)
        sim.apply_link_perf_scaling(efficient_nodes=eff)

        res=sim.run_once(total_ticks=ticks_total, pairs_per_tick=args.pairs,
                         sleep_strategy='none', log_progress=args.log_progress,
                         use_throughput_guard=False, consolidated=False)
        rows2.append([pct, res["total_energy_Wh"], res["avg_latency_ms"], res["p50_latency_ms"],
                      res["p95_latency_ms"], res["total_drops"]])

    # Charts for upgrade sweep
    write_csv("out/chart4_upgrade_energy.csv",
              ["upgrade_percent","total_energy_Wh","avg_latency_ms","p50_latency_ms","p95_latency_ms","total_drops"],
              rows2)
    write_csv("out/chart5_upgrade_latency.csv",
              ["upgrade_percent","avg_latency_ms","p50_latency_ms","p95_latency_ms"],
              [[r[0], r[2], r[3], r[4]] for r in rows2])
    write_csv("out/chart6_upgrade_drops.csv", ["upgrade_percent","total_drops"], [[r[0], r[5]] for r in rows2])

    # Run manifest for reproducibility
    with open("out/run_summary.json","w") as f:
        json.dump({"args":vars(args),
                   "notes":"Weekly profile; consolidation; adaptive/fixed guards; timeseries optional; p50/p95 reporting; ops summaries per sleep cap."}, f, indent=2)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    """
    CLI for the simulator. Defaults target a 7-day run with 5-min ticks, 
    scale-free topology, adaptive sleeping, and adaptive guard.
    """
    ap=argparse.ArgumentParser(description="SDN Sleep vs Hardware-Upgrade Energy Simulator (weekly + adaptive guard + ops)")
    # Horizon & time base
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--tick-min", type=int, default=5)
    # Topology & traffic (scale_free is ISP/WAN-like default)
    ap.add_argument("--topology", choices=["scale_free","fat_tree"], default="scale_free")
    ap.add_argument("--policy", choices=["static","adaptive","none"], default="adaptive")
    ap.add_argument("--pairs", type=int, default=160)
    ap.add_argument("--burst-shape", type=float, default=2.0)
    ap.add_argument("--burst-scale", type=float, default=0.02)
    ap.add_argument("--heavy-user-share", type=float, default=0.015)
    # Weekly profile files
    # Default now points to the /in/ subfolder per your input-staging convention.
    ap.add_argument("--traffic-profile-csv", type=str, default="in/traffic_profile_us.csv")
    ap.add_argument("--res-share", type=float, default=0.85)
    ap.add_argument("--event-csv", type=str, default="")
    # Energy model
    ap.add_argument("--hw-profile", choices=["none","std_qfx5120","eff_arista_7050x"], default="std_qfx5120")
    ap.add_argument("--P-idle-std", dest="P_idle_std", type=float, default=210.0)
    ap.add_argument("--P-dyn-std",  dest="P_dyn_std",  type=float, default=60.0)
    ap.add_argument("--efficient-idle-factor", type=float, default=0.6)
    ap.add_argument("--efficient-dyn-factor",  type=float, default=0.6)
    ap.add_argument("--P-sleep", dest="P_sleep", type=float, default=10.0)
    ap.add_argument("--t-wake-ms", type=int, default=20)
    ap.add_argument("--node-capacity-gbps-proxy", type=float, default=20.0)
    # Link EEE
    ap.add_argument("--P-link-idle", dest="P_link_idle", type=float, default=2.0)
    ap.add_argument("--P-link-sleep", dest="P_link_sleep", type=float, default=0.1)
    ap.add_argument("--t-link-wake-ms", type=int, default=2)
    ap.add_argument("--link-sleep-rho", type=float, default=0.05)
    # Connectivity & SLO
    ap.add_argument("--avoid-articulation", action="store_true", default=True)
    ap.add_argument("--sleep-hysteresis", type=int, default=5)
    ap.add_argument("--link-hysteresis", type=int, default=5)
    ap.add_argument("--slo-window", type=int, default=30)
    ap.add_argument("--max-drop-rate", type=float, default=0.02)
    ap.add_argument("--max-latency-ms", type=float, default=50.0)
    # Guards
    ap.add_argument("--guard-mode", choices=["fixed","adaptive"], default="adaptive")
    ap.add_argument("--throughput-guard", action="store_true", default=True)
    ap.add_argument("--throughput-margin", type=float, default=0.70, help="used only for fixed guard")
    ap.add_argument("--rho-target", type=float, default=0.60, help="adaptive guard: target max link utilization")
    ap.add_argument("--headroom", type=float, default=1.20, help="adaptive guard: safety margin multiplier")
    ap.add_argument("--spine-frac", type=float, default=0.18, help="fraction of high-degree nodes pinned on")
    # Perf scaling near upgraded nodes
    ap.add_argument("--cap-scale-eff", type=float, default=1.15)
    ap.add_argument("--lat-scale-eff", type=float, default=0.95)
    # Trace (Milestone 2)
    ap.add_argument("--trace-csv", type=str, default="")
    ap.add_argument("--node-map-csv", type=str, default="")
    ap.add_argument("--trace-scale", type=float, default=1.0)
    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-progress", action="store_true")
    ap.add_argument("--emit-timeseries", action="store_true")
    # legacy diurnal (unused when weekly profile applies)
    ap.add_argument("--peak-hour", type=float, default=18.0)
    ap.add_argument("--min-load", type=float, default=0.3)
    return ap.parse_args()


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__=="__main__":
    args=parse_args()
    args = apply_hw_profile(args)
    run_experiments(args)

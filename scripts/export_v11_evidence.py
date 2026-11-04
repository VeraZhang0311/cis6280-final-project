"""Export a Git-sized G1-G7 evidence bundle, or verify it without cluster/output access."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil


RUNS={
    'bidir-1498961':dict(snapshot='cis6280-v11-bidir-r2',workers=8,gpus=4,
        files=['run.json','summary.json','selection.json','execution.log','analysis/episodes.csv','analysis/paired-cases.csv',
               'analysis/representative-cases.json','analysis/local-evidence-check.json',
               'analysis/outcomes.png','analysis/paired-cases.png','analysis/alignment-mechanism.png','analysis/pilot-outcomes.png']),
    'sampling-1499337':dict(snapshot='cis6280-v11-sampling-r3',workers=4,gpus=2,
        files=['run.json','summary.json','execution.log','analysis/episodes.csv','analysis/paired-cases.csv',
               'analysis/representative-cases.json','analysis/local-evidence-check.json',
               'analysis/task-and-cost.png','analysis/budgets-and-pairs.png']),
}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def cases_table(summary, sampling=False):
    rows=['| 组别 | mean peak | mean final | ≥2 箱 | ≥3 箱 | 曾达4箱 / success | 平均 / P95 决策 ms |'+(' 平均候选数 |' if sampling else ''),
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'+(' ---: |' if sampling else '')]
    for g,s in summary['full'].items():
        n=s['episodes']
        count=lambda k:round(s['reached'][f'placed>={k}']*n)
        rows.append(f'| {g} | {s["mean_max_placed"]:.3f} | {s["mean_final_count"]:.3f} | {count(2)}/{n} | {count(3)}/{n} | '
                    f'{count(4)}/{n} / {round(s["success"]*n)}/{n} | {s["decision_s_mean"]*1000:.1f} / {s["decision_s_p95"]*1000:.1f} |'+
                    (f' {s["sampling"]["mean_candidates"]:.2f} |' if sampling else ''))
    return '\n'.join(rows)


def pair_rows(label,pairs):
    rows=[]
    for name,p in pairs.items():
        lo,hi=p['mean_peak_delta_ci95_scene_bootstrap']
        rows.append(f'| {label} {name} | {p["mean_peak_delta"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] | '
                    f'{p["improved"]} / {p["tied"]} / {p["regressed"]} |')
    return rows


def export(repo):
    out=repo/'notes/v1.1';bundle=out/'evidence';bundle.mkdir(parents=True,exist_ok=True)
    artifacts=[];cohorts={};records=[];index=[];summaries={}
    for run_id,spec in RUNS.items():
        root=repo/'outputs/planning/v1.1'/run_id
        run,summary=read(root/'run.json'),read(root/'summary.json')
        if run['status']!='complete':raise ValueError(f'{run_id} is not complete')
        p=run['protocol']
        if p['workers']!=spec['workers'] or len(p['gpus'].split(','))!=spec['gpus']:
            raise ValueError('unexpected execution settings')
        summaries[run_id]=summary
        cluster=f'/mnt/projects/ll/tianzizh/{spec["snapshot"]}/outputs/planning/v1.1/{run_id}'
        counts={}
        for source_name in spec['files']:
            source=root/source_name
            dest=bundle/run_id/Path(source_name).name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest);artifacts.append(dest)
        for stage in ('full','pilot'):
            if not (root/stage).exists():continue
            counts[stage]={}
            for group in sorted((root/stage).glob('G*')):
                markers=sorted((group/'episodes').glob('*/complete.json'))
                expected=60 if stage=='full' else 12
                if len(markers)!=expected:raise ValueError(f'incomplete {stage}/{group.name}')
                dest=bundle/run_id/'protocols'/f'{stage}-{group.name}.json'
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(group/'run.json',dest);artifacts.append(dest)
                counts[stage][group.name]=len(markers)
                for marker in markers:
                    ep=marker.parent;result=read(ep/'summary.json')
                    relative=ep.relative_to(root).as_posix()
                    key=f'{run_id}/{relative}'
                    records.append(dict(evidence_id=key,cohort=run_id,stage=stage,variant=group.name,result=result))
                    files={f:identity(ep/f) for f in ('complete.json','summary.json','steps.jsonl','events.jsonl',
                        'video.mp4','timeline.png','tail-state.json','objects.json')}
                    keyframes=sorted(x.name for x in (ep/'keyframes').glob('*.png'))
                    if len(keyframes)!=6:raise ValueError('missing keyframes')
                    index.append(dict(evidence_id=key,cohort=run_id,relative_path=relative,
                                      files=files,keyframes=keyframes))
        cohorts[run_id]=dict(protocol_version=p['version'],protocol_id=run['protocol_id'],status=run['status'],
            source_code=p['code'],inputs=p['inputs'],workers=p['workers'],gpus=p['gpus'],runtime=run['runtime'],
            scenes=p['full_scenes'],sample_seeds=p['sample_seeds'],task=p['task'],counts=counts,
            local_root=root.relative_to(repo).as_posix(),cluster_root=cluster,
            started_utc=run['started_utc'],ended_utc=run['ended_utc'])
    a,b=(summaries[k] for k in RUNS)
    if cohorts['bidir-1498961']['scenes']!=cohorts['sampling-1499337']['scenes']:
        raise ValueError('cohort scenes differ')
    for name in ('episode-results','evidence-index'):
        path=bundle/f'{name}.jsonl'
        rows=records if name=='episode-results' else index
        path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8',newline='\n')
        artifacts.append(path)
    g1,g6,g7=(b['full'][g] for g in ('G1','G6','G7'))
    pairs=pair_rows('A',{f'{g}-G0':p for g,p in a['paired'].items()})+pair_rows('A',a['contrasts'])+pair_rows('B',b['paired'])
    definitions='''| 组别 | 目标切换 | 候选评分 / 预算 |
| --- | --- | --- |
| G0（辅助对照） | 官方规则 | 第16个预测 latent 距离，16候选 |
| G1 | 官方规则 | 第13–16步距离最小值，16候选 |
| G2 | 官方规则 | 第13–16步按1:2:3:4加权，16候选 |
| G3 | 停滞后单向向前对齐 | 终点评分，16候选 |
| G4 | 停滞后双向对齐，允许回退 | 终点评分，16候选 |
| G5 | 与新G4相同的双向对齐 | pilot锁定的G2加权评分，16候选 |
| G6 | 与G1相同，官方规则 | G1尾窗最小评分；16→32→最多48，自适应补采 |
| G7 | 与G1相同，官方规则 | G1尾窗最小评分；一次采样48条 |'''
    text=f'''# G1–G7 实验总结与证据入口

更新日期：2026-11-04。两批实验均已完成并同步本地。**先按批次看结果，再比较方法；两批 G1 不是同一份对照。**

## 1. 主要结果

- A批（G0–G5）：G4 的 mean peak 最高（1.667），G1 的 mean final 最高（1.383）；五个处理组对 G0 的 mean peak 差异，其95%区间均包含0。G5 的 mean peak 低于 G4，但 mean final 稍高（1.267 vs 1.233）。
- B批（G1/G6/G7）：G7 的 mean peak/final 最高（1.717/1.433），G1 达到至少3箱的案例更多（12/60，对比G6/G7各10/60）。
- G6 平均每轮 {g6['sampling']['mean_candidates']:.2f} 条候选，G7 固定48条；两者平均决策耗时却几乎一样（{g6['decision_s_mean']*1000:.1f} vs {g7['decision_s_mean']*1000:.1f} ms）。G7 的P95低约 {(1-g7['decision_s_p95']/g6['decision_s_p95'])*100:.1f}%。
- G7 相对同批G1的 mean peak 增加0.250，平均决策耗时增加约 {(g7['decision_s_mean']/g1['decision_s_mean']-1)*100:.1f}%；G7对G1、G6的配对 mean peak 区间都包含0，当前排序不足以证明稳定提升。

## 2. 方法定义与共同设置

{definitions}

共同设置：冻结官方 policy、encoder、WM；16个预测blocks（8秒）；每个block为5个control steps（0.5秒）；只执行第一个block；n_steps=10、lag=0、tau=0.25。相机分辨率128，固定20个scene、sampler seeds 0/2/3；episode seed为 `scene + 1000000 * sample_seed`。目标图片仍为官方每8秒序列，距离为未另行归一化latent的平方L2。候选评分越小越好。

G3/G4/G5 的停滞条件为当前目标停留24秒，最近8秒首末各4次真实距离的中位数下降不足10%；未来/附近目标须满足 `q=D/T<1` 并连续确认3次。G4/G5 搜索当前索引±3，允许回退；重对齐当轮直接使用匹配目标，下一次真实观测才恢复官方推进。三组均无 timeout 强制前移。

G6 在累计16/32条后，仅当 `C_best>=T` 且 `r=(D_now-C_best)/max(D_now,1e-8)<0.20` 才补采；近零D_now不补采，48条无条件结束。保留较早最佳，所有批次使用同一真实上下文，额外批次只rollout新候选。G7只有一次48条的采样/rollout。不同batch形状可能改变RNG/数值结果，不要求两组候选前缀相同。

**命名沿革：旧协议 `run-1498776` 的G4是单向对齐组合组；本总结的G4是双向终点评分，G5才是双向组合组。旧G4未改名充当新组，也未纳入以下统计。** 历史G5使用G2，后续选择G1作为G6/G7基础不改变它。

## 3. 两个可比较的批次

| 批次 | 协议 / 输出ID | 全量 | GPU / workers | 用途 |
| --- | --- | --- | --- | --- |
| A | v1.1-bidirectional-r2 / bidir-1498961 | G0–G5各60，共360 | 4×2080 Ti / 8 | 评分、单向、双向及组合对照 |
| B | v1.1-sampling-r3 / sampling-1499337 | G1/G6/G7各60，共180 | 2×2080 Ti / 4 | 自适应补采与固定48对照；无pilot |

两批均使用FP16 encoder/WM autocast，policy为FP32且无policy autocast；依赖见各自run.json。A另有72个pilot记录；B没有pilot。A的兼容旧G0–G3记录按原协议/文件哈希复用，出处保留在manifest。B因分配变化重跑完整G1，**没有把A的G1结果混入B**。G1来自已看过的开发结果，全部实验仍使用同一20个开发场景，不是独立测试集。

旧L40/BF16的Phase 1、单场景A10、43窗口open-loop诊断，以及更早旧G4，不并入这两张表。

### A：G0–G5，每组60条

{cases_table(a)}

![A批任务与配对结果](evidence/bidir-1498961/outcomes.png)

![A批逐scene/seed的peak差异](evidence/bidir-1498961/paired-cases.png)

### B：同配置重跑的G1、G6、G7，每组60条

{cases_table(b,True)}

![B批任务、候选数与耗时](evidence/sampling-1499337/task-and-cost.png)

![B批预算分布和配对差异](evidence/sampling-1499337/budgets-and-pairs.png)

决策耗时是每次 `agent.step` 前后GPU同步后的wall time，包含编码、采样、rollout、评分及agent内诊断；不包含仿真control loop、视频编码或传输。mean/P95基于该组全部决策，不能把180个episode总耗时直接当成单次决策耗时。三组在各自同批使用相同worker与视频设置；共享节点负载仍可能影响计时。

## 4. 配对结果与不确定性

正号表示左侧组mean peak更高。按相同scene/seed配对；bootstrap以整个scene（保留3个sampler seeds）为单位，5000次、seed=1103。不是把60条当成60个独立场景。区间为开发性比较，未做多重比较校正。

| 同批比较 | mean peak差异 | 95% scene-bootstrap区间 | 改善 / 持平 / 退步（60条） |
| --- | ---: | --- | ---: |
{chr(10).join(pairs)}

G5相对较弱G2的区间高于0，但它对G0/G4的区间仍跨0；不能据此宣称组合方案普遍更好。原始配对case表和区间都已归档，无需依赖聊天文字复算。

## 5. G6预算究竟花在哪里

| 实际候选数 | 决策轮数 | 比例 |
| --- | ---: | ---: |
'''
    sample=g6['sampling'];decisions=sample['decisions']
    for k,n in sample['candidate_count_distribution'].items():text+=f'| {k} | {n} | {n/decisions:.2%} |\n'
    selected=sample['selected_batch_distribution']
    text+=f'''
G6 共{decisions}次决策，第二批补采率{sample['second_batch_rate']:.2%}，第三批补采率{sample['third_batch_rate']:.2%}；最终赢家来自第1/2/3批的轮数为 {selected.get('1',0)} / {selected.get('2',0)} / {selected.get('3',0)}。平均每轮policy和WM各调用{sample['policy_sample_calls_mean']:.3f}次，G7各1次。两组最大候选预算相同，不代表实际计算量相同。

G6少采候选没有转化为本批平均耗时下降，这与多批串行调用/批处理效率相符；没有单独GPU profiler实验，不能将耗时差异全部归因于某一个原因。每批cost改善和停止原因保存在原始summary/steps中；这些是机制证据，不是任务成功率。

## 6. 计数与诊断的解释

- peak：每个control step观测到的最大有效tower箱数；final：episode结束的有效箱数。目标区内互不相连的结构不相加；手接触等条件会影响有效计数。计数下降不自动等于倒塌。
- “曾达4箱”与 `success` 不相同：官方success还要求目标计数达标且手完全释放，连续保持10个control steps。表中两项分列，避免把瞬时peak当成正式成功。
- 到第二箱的成功者耗时不代表所有episode；未完成者仍保留在episodes.csv和episode-results.jsonl中，包括第一箱后剩余观察时间。
- goal distance是平方L2；预测MSE是上一轮第一个预测latent与下一次真实latent的逐维均方差；plan change RMS是本轮首block与上轮第二block在裁剪后的raw action单位中的RMS差。
- 不同active goal的原始距离不可连成统一改善曲线，目标索引更快或发生回退本身都不是任务收益。G4/G5需要结合切换日志和真实视频查看往返行为。

## 7. 已纳入Git的证据包

入口：[evidence/manifest.json](evidence/manifest.json)。该包可在没有outputs目录、没有GPU和SSH的clone中查看。

| 文件 | 内容 |
| --- | --- |
| evidence/manifest.json | 两批身份、实际输入/源码哈希、场景/seed、GPU/dtype、来源和归档文件SHA256 |
| evidence/episode-results.jsonl | 612条完整episode summary：540条full + 72条pilot，带批次和stage标签 |
| evidence/evidence-index.jsonl | 612条原始证据路径与文件SHA256；包含视频、steps/events、tail及关键帧数量 |
| evidence/各run-id/run.json、summary.json | 按原字节复制的最终manifest和全部统计/配对case；不改写历史文件 |
| evidence/各run-id/execution.log | 原始Slurm输出，含运行前测试结果及逐episode完成记录 |
| evidence/各run-id/protocols/ | 每个stage/组实际配置；A的selection.json保留G5 pilot选择依据 |
| evidence/各run-id/episodes.csv、paired-cases.csv | episode与配对平表，保留失败和censored观察 |
| evidence/各run-id/*.png | 本文展示的图、机制图及pilot图（有明确stage，未混入full） |
| evidence/各run-id/representative-cases.json | 原报告确定性选出的改善/持平/退步案例索引；不是人工行为标注 |

原始视频和大体积steps/tail文件保留在本地/cluster，不塞进Git。全量日志没有保存每轮全部候选动作或完整latent张量。模型/目标文件未提交，但manifest提供三个输入SHA256。实验采用运行时的未提交源码快照；manifest保留当时的commit基点和实际Python源码哈希，不能声称它们运行于本次新commit。

## 8. 打开视频、取回原始证据与复现

本机已有全部数据（以下链接依赖本地outputs，不是Git仓库内文件）：

- [A批完整视频报告](../../outputs/planning/v1.1/bidir-1498961/analysis/index.html)
- [B批完整视频报告](../../outputs/planning/v1.1/sampling-1499337/analysis/index.html)

任意episode目录含 `video.mp4`、`steps.jsonl`、`events.jsonl`、`summary.json`、`timeline.png`、`objects.json`、`tail-state.json`、6张 `keyframes/*.png` 和 `complete.json`。报告提供代表案例并列播放，按对应目录可查全部失败案例。源目录见manifest的local_root/cluster_root；证据索引的relative_path相对对应根目录。

```powershell
New-Item -ItemType Directory -Force outputs/planning/v1.1 | Out-Null
scp -O -r tianzizh@grasp-login2.seas.upenn.edu:/mnt/projects/ll/tianzizh/cis6280-v11-bidir-r2/outputs/planning/v1.1/bidir-1498961 outputs/planning/v1.1/
scp -O -r tianzizh@grasp-login2.seas.upenn.edu:/mnt/projects/ll/tianzizh/cis6280-v11-sampling-r3/outputs/planning/v1.1/sampling-1499337 outputs/planning/v1.1/
```

证据包的完整性检查只需Python标准库，仓库clone后即可执行：

```text
python scripts/export_v11_evidence.py --verify
```

当上述两个原始目录齐全时，重新生成本文和证据包：

```text
python scripts/export_v11_evidence.py
```

实现入口：[agent.py](../../assignment/training/planning/agent.py)、[v11.py](../../assignment/training/planning/v11.py)、[v11_sampling.py](../../assignment/training/planning/v11_sampling.py)。
详细协议：[DESIGN.md](DESIGN.md)；实现与验证：[IMPLEMENTATION.md](IMPLEMENTATION.md)；完整运行沿革与cluster命令：[RUN.md](RUN.md)。34项关键测试与G0–G5各100次决策的旧快照等价检查已通过；测试日志原件位于对应cluster的Slurm日志。重跑命令/冻结源码目录在RUN.md中，需准备同SHA256输入和相应GPU环境；不保证跨硬件/依赖逐帧轨迹一致。
'''
    report=out/'RESULTS.md';report.write_text(text,encoding='utf-8',newline='\n');artifacts.append(report)
    manifest=dict(schema_version=1,description='G1-G7 full development evidence; cohorts and pilots stay distinct',
        cohorts=cohorts,total_full=540,total_pilot=72,total_indexed=612,
        artifacts={path.relative_to(repo).as_posix():identity(path) for path in sorted(artifacts)})
    write(bundle/'manifest.json',manifest)
    return verify(repo)


def verify(repo):
    bundle=repo/'notes/v1.1/evidence';manifest=read(bundle/'manifest.json')
    for relative,expected in manifest['artifacts'].items():
        if identity(repo/relative)!=expected:raise ValueError(f'archive hash mismatch: {relative}')
    records=[json.loads(s) for s in (bundle/'episode-results.jsonl').read_text(encoding='utf-8').splitlines()]
    indexed=[json.loads(s) for s in (bundle/'evidence-index.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(records)!=612 or len(indexed)!=612 or len({r['evidence_id'] for r in records})!=612 or {r['evidence_id'] for r in records}!={r['evidence_id'] for r in indexed}:
        raise ValueError('evidence case identities/counts differ')
    counts=Counter((r['cohort'],r['stage'],r['variant']) for r in records)
    for (cohort,stage,group),n in counts.items():
        if n!=(60 if stage=='full' else 12):raise ValueError('case count mismatch')
        summary=read(bundle/cohort/'summary.json')[stage][group]
        results=[r['result'] for r in records if (r['cohort'],r['stage'],r['variant'])==(cohort,stage,group)]
        for field,metric in (('max_placed','mean_max_placed'),('final_count','mean_final_count')):
            if not math.isclose(sum(r[field] for r in results)/n,summary[metric],abs_tol=1e-12):
                raise ValueError('summary does not match per-episode results')
        if not math.isclose(sum(bool(r['success']) for r in results)/n,summary['success'],abs_tol=1e-12):
            raise ValueError('success does not match per-episode results')
        for k in range(1,5):
            if not math.isclose(sum(r['max_placed']>=k for r in results)/n,summary['reached'][f'placed>={k}'],abs_tol=1e-12):
                raise ValueError('reach fraction does not match per-episode results')
        weighted_time=sum(r['decision_s_mean']*r['decision_count'] for r in results)/sum(r['decision_count'] for r in results)
        if not math.isclose(weighted_time,summary['decision_s_mean'],abs_tol=1e-12):
            raise ValueError('mean decision latency differs from weighted episode means')
    for cohort in RUNS:
        summary=read(bundle/cohort/'summary.json')
        by_group={g:{(r['result']['scene_seed'],r['result']['sample_seed']):r['result'] for r in records
                     if (r['cohort'],r['stage'],r['variant'])==(cohort,'full',g)} for g in summary['full']}
        pairs={f'{g}-G0':p for g,p in summary['paired'].items()} if cohort=='bidir-1498961' else dict(summary['paired'])
        pairs.update(summary.get('contrasts',{}))
        for label,pair in pairs.items():
            treatment,baseline=label.split('-');a,b=by_group[treatment],by_group[baseline]
            deltas=[a[k]['max_placed']-b[k]['max_placed'] for k in sorted(a)]
            if set(a)!=set(b) or len(deltas)!=60 or not math.isclose(sum(deltas)/60,pair['mean_peak_delta'],abs_tol=1e-12):
                raise ValueError('paired mean does not match episode identities/outcomes')
            if [sum(d>0 for d in deltas),sum(d==0 for d in deltas),sum(d<0 for d in deltas)]!=[pair[k] for k in ('improved','tied','regressed')]:
                raise ValueError('paired counts differ from episode outcomes')
    print(f'Verified {len(manifest["artifacts"])} archived files; 540 full + 72 pilot records and indexed raw evidence; all task aggregates agree.')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    verify(args.project_root.resolve()) if args.verify else export(args.project_root.resolve())

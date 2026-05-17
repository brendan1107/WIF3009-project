import { useEffect, useMemo, useState } from 'react'
import championData from './assets/data/champions.json'
import './App.css'

type Role = 'TOP' | 'JUNGLE' | 'MID' | 'BOTTOM' | 'SUPPORT'
type Team = 'blue' | 'red'
type DraftAction = 'ban' | 'pick'
type SortKey = 'winRate' | 'name' | 'synergy' | 'counter'

type Champion = {
  id: string
  key: string
  name: string
  title: string
  tags: string[]
  image: string
  info: {
    attack: number
    defense: number
    magic: number
    difficulty: number
  }
  stats: {
    hp: number
    armor: number
    spellblock: number
    attackdamage: number
    attackrange: number
    movespeed: number
  }
}

type Slot = {
  role: Role
  player: string
  championId?: string
}

type DraftStep = {
  phase: 'Ban Phase 1' | 'Pick Phase 1' | 'Ban Phase 2' | 'Pick Phase 2'
  action: DraftAction
  team: Team
  label: string
  banIndex?: number
  slotIndex?: number
}

type DraftPayload = {
  bluePicks: Slot[]
  redPicks: Slot[]
  blueBans: string[]
  redBans: string[]
  activeTeam: Team
  activeAction: DraftAction
  activeRole?: Role
  stepIndex: number
}

type Prediction = {
  blueWinRate: number
  redWinRate: number
  source: 'backend' | 'local'
}

type LockFeedback = {
  action: DraftAction
  banIndex?: number
  championName: string
  slotIndex?: number
  stamp: number
  team: Team
}

const championIcons = import.meta.glob('./assets/champion/*.png', {
  eager: true,
  import: 'default',
  query: '?url',
}) as Record<string, string>

const roles: Role[] = ['TOP', 'JUNGLE', 'MID', 'BOTTOM', 'SUPPORT']

const draftSteps: DraftStep[] = [
  { phase: 'Ban Phase 1', action: 'ban', team: 'blue', banIndex: 0, label: 'Blue Ban 1' },
  { phase: 'Ban Phase 1', action: 'ban', team: 'red', banIndex: 0, label: 'Red Ban 1' },
  { phase: 'Ban Phase 1', action: 'ban', team: 'blue', banIndex: 1, label: 'Blue Ban 2' },
  { phase: 'Ban Phase 1', action: 'ban', team: 'red', banIndex: 1, label: 'Red Ban 2' },
  { phase: 'Ban Phase 1', action: 'ban', team: 'blue', banIndex: 2, label: 'Blue Ban 3' },
  { phase: 'Ban Phase 1', action: 'ban', team: 'red', banIndex: 2, label: 'Red Ban 3' },
  { phase: 'Pick Phase 1', action: 'pick', team: 'blue', slotIndex: 0, label: 'Blue Pick 1' },
  { phase: 'Pick Phase 1', action: 'pick', team: 'red', slotIndex: 0, label: 'Red Pick 1' },
  { phase: 'Pick Phase 1', action: 'pick', team: 'red', slotIndex: 1, label: 'Red Pick 2' },
  { phase: 'Pick Phase 1', action: 'pick', team: 'blue', slotIndex: 1, label: 'Blue Pick 2' },
  { phase: 'Pick Phase 1', action: 'pick', team: 'blue', slotIndex: 2, label: 'Blue Pick 3' },
  { phase: 'Pick Phase 1', action: 'pick', team: 'red', slotIndex: 2, label: 'Red Pick 3' },
  { phase: 'Ban Phase 2', action: 'ban', team: 'red', banIndex: 3, label: 'Red Ban 4' },
  { phase: 'Ban Phase 2', action: 'ban', team: 'blue', banIndex: 3, label: 'Blue Ban 4' },
  { phase: 'Ban Phase 2', action: 'ban', team: 'red', banIndex: 4, label: 'Red Ban 5' },
  { phase: 'Ban Phase 2', action: 'ban', team: 'blue', banIndex: 4, label: 'Blue Ban 5' },
  { phase: 'Pick Phase 2', action: 'pick', team: 'red', slotIndex: 3, label: 'Red Pick 4' },
  { phase: 'Pick Phase 2', action: 'pick', team: 'blue', slotIndex: 3, label: 'Blue Pick 4' },
  { phase: 'Pick Phase 2', action: 'pick', team: 'blue', slotIndex: 4, label: 'Blue Pick 5' },
  { phase: 'Pick Phase 2', action: 'pick', team: 'red', slotIndex: 4, label: 'Red Pick 5' },
]

const roleProfiles: Record<Role, Partial<Champion['info']>> = {
  TOP: { attack: 7, defense: 7 },
  JUNGLE: { attack: 7, defense: 5 },
  MID: { magic: 8, attack: 5 },
  BOTTOM: { attack: 8 },
  SUPPORT: { defense: 7, magic: 6 },
}

const roleOverrides: Record<string, Role[]> = {
  Ahri: ['MID'],
  Akali: ['MID', 'TOP'],
  Anivia: ['MID'],
  Cassiopeia: ['MID'],
  Darius: ['TOP'],
  Jhin: ['BOTTOM'],
  Katarina: ['MID'],
  Leblanc: ['MID'],
  LeeSin: ['JUNGLE'],
  Lissandra: ['MID'],
  Lux: ['MID', 'SUPPORT'],
  Orianna: ['MID'],
  Ornn: ['TOP'],
  Samira: ['BOTTOM'],
  Sylas: ['MID', 'TOP'],
  Syndra: ['MID'],
  Taliyah: ['MID', 'JUNGLE'],
  Veigar: ['MID'],
  Vi: ['JUNGLE'],
  Viktor: ['MID'],
  Yasuo: ['MID', 'TOP'],
  Zed: ['MID'],
}

const positionIcon: Record<Role, string> = {
  TOP: 'T',
  JUNGLE: 'J',
  MID: 'M',
  BOTTOM: 'B',
  SUPPORT: 'S',
}

const initialBlueSlots: Slot[] = [
  { role: 'MID', player: 'Player 1' },
  { role: 'JUNGLE', player: 'Player 2' },
  { role: 'TOP', player: 'Player 3' },
  { role: 'SUPPORT', player: 'Player 4' },
  { role: 'BOTTOM', player: 'Player 5' },
]

const initialRedSlots: Slot[] = [
  { role: 'JUNGLE', player: 'Enemy 1' },
  { role: 'TOP', player: 'Enemy 2' },
  { role: 'BOTTOM', player: 'Enemy 3' },
  { role: 'MID', player: 'Enemy 4' },
  { role: 'SUPPORT', player: 'Enemy 5' },
]

const emptyBans = ['', '', '', '', '']

const champions = (championData.champions as Champion[]).filter(
  (champion) => championIcons[`./assets/champion/${champion.id}.png`],
)

function championById(id?: string) {
  return champions.find((champion) => champion.id === id)
}

function iconFor(id?: string) {
  if (!id) return undefined
  return championIcons[`./assets/champion/${id}.png`]
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

function rolesForChampion(champion: Champion): Role[] {
  const override = roleOverrides[champion.id]
  if (override) return override

  const inferred = new Set<Role>()
  if (champion.tags.includes('Tank') || champion.tags.includes('Fighter')) {
    inferred.add('TOP')
  }
  if (champion.tags.includes('Assassin') || champion.tags.includes('Fighter')) {
    inferred.add('JUNGLE')
  }
  if (champion.tags.includes('Mage') || champion.tags.includes('Assassin')) {
    inferred.add('MID')
  }
  if (champion.tags.includes('Marksman')) {
    inferred.add('BOTTOM')
  }
  if (champion.tags.includes('Support')) {
    inferred.add('SUPPORT')
  }
  return inferred.size ? [...inferred] : ['MID']
}

function roleFit(champion: Champion, role: Role) {
  const roleList = rolesForChampion(champion)
  const profile = roleProfiles[role]
  const statFit =
    (profile.attack ? 10 - Math.abs(champion.info.attack - profile.attack) : 0) +
    (profile.defense ? 10 - Math.abs(champion.info.defense - profile.defense) : 0) +
    (profile.magic ? 10 - Math.abs(champion.info.magic - profile.magic) : 0)

  return (roleList.includes(role) ? 12 : 0) + statFit / 2
}

function championMetrics(champion: Champion, role: Role) {
  const base =
    48 +
    roleFit(champion, role) * 0.32 +
    champion.info.attack * 0.12 +
    champion.info.magic * 0.1 +
    champion.info.defense * 0.08 -
    champion.info.difficulty * 0.05
  const keyNoise = (Number(champion.key) % 17) / 10
  const winRate = clamp(base + keyNoise, 45.5, 53.4)
  const synergy = Math.round(
    clamp(58 + roleFit(champion, role) * 2 + champion.info.defense * 1.8, 40, 94),
  )
  const counter = Math.round(
    clamp(42 + champion.info.attack * 2 + champion.info.magic * 1.3, 35, 91),
  )
  const sampleSize = 8700 + (Number(champion.key) % 8500)

  return { winRate, synergy, counter, sampleSize }
}

function teamScore(slots: Slot[]) {
  return slots.reduce((score, slot) => {
    const champion = championById(slot.championId)
    if (!champion) return score
    const metrics = championMetrics(champion, slot.role)
    return score + metrics.winRate + metrics.synergy * 0.08 + metrics.counter * 0.04
  }, 0)
}

function localPrediction(payload: DraftPayload): Prediction {
  const blueScore = teamScore(payload.bluePicks)
  const redScore = teamScore(payload.redPicks)
  const blueFilled = payload.bluePicks.filter((slot) => slot.championId).length
  const redFilled = payload.redPicks.filter((slot) => slot.championId).length
  const blueBans = payload.blueBans.filter(Boolean).length
  const redBans = payload.redBans.filter(Boolean).length
  const banPressure = (redBans - blueBans) * 0.18
  const activeBonus = payload.activeTeam === 'blue' ? 0.35 : -0.35
  const blueWinRate = clamp(
    50 + (blueScore - redScore) * 0.16 + (blueFilled - redFilled) * 0.72 + banPressure + activeBonus,
    38,
    62,
  )

  return {
    blueWinRate: Number(blueWinRate.toFixed(1)),
    redWinRate: Number((100 - blueWinRate).toFixed(1)),
    source: 'local',
  }
}

async function requestPrediction(payload: DraftPayload): Promise<Prediction> {
  const endpoint = import.meta.env.VITE_WINRATE_API_URL
  if (!endpoint) return localPrediction(payload)

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!response.ok) {
      throw new Error(`Prediction API returned ${response.status}`)
    }

    const result = (await response.json()) as { blueWinRate: number; redWinRate?: number }
    return {
      blueWinRate: Number(result.blueWinRate.toFixed(1)),
      redWinRate: Number((result.redWinRate ?? 100 - result.blueWinRate).toFixed(1)),
      source: 'backend',
    }
  } catch {
    return localPrediction(payload)
  }
}

function teamName(team: Team) {
  return team === 'blue' ? 'Blue Team' : 'Red Team'
}

function currentPickRole(step: DraftStep | undefined, blueSlots: Slot[], redSlots: Slot[]) {
  if (!step || step.action !== 'pick' || step.slotIndex === undefined) return undefined
  return step.team === 'blue' ? blueSlots[step.slotIndex]?.role : redSlots[step.slotIndex]?.role
}

function nextPickRole(stepIndex: number, blueSlots: Slot[], redSlots: Slot[]) {
  const nextPick = draftSteps.slice(stepIndex).find((step) => step.action === 'pick')
  return currentPickRole(nextPick, blueSlots, redSlots) ?? 'MID'
}

function TeamMark({ team }: { team: Team }) {
  return (
    <div className={`team-mark ${team}`} aria-hidden="true">
      <span />
      <span />
      <span />
    </div>
  )
}

function EmptyPortrait() {
  return (
    <div className="empty-portrait" aria-hidden="true">
      <span />
    </div>
  )
}

function ChampionPortrait({ id, alt }: { id?: string; alt: string }) {
  const src = iconFor(id)
  if (!src) return <EmptyPortrait />
  return <img className="champion-portrait" src={src} alt={alt} />
}

function TeamPanel({
  activeBanIndex,
  activeSlotIndex,
  activeTeam,
  bans,
  feedback,
  slots,
  team,
}: {
  activeBanIndex?: number
  activeSlotIndex?: number
  activeTeam: Team
  bans: string[]
  feedback: LockFeedback | null
  slots: Slot[]
  team: Team
}) {
  return (
    <aside className={`team-panel ${team}`}>
      <h2>{teamName(team)}</h2>
      <div className="slot-list">
        {slots.map((slot, index) => {
          const champion = championById(slot.championId)
          const isActive = team === activeTeam && index === activeSlotIndex
          const isLocked =
            feedback?.action === 'pick' &&
            feedback.team === team &&
            feedback.slotIndex === index

          return (
            <button
              className={`draft-slot ${isActive ? 'active' : ''} ${isLocked ? 'locked' : ''}`}
              key={`${slot.player}-${slot.role}`}
              type="button"
            >
              <span className="role-sigil">{positionIcon[slot.role]}</span>
              <ChampionPortrait id={slot.championId} alt={champion?.name ?? `${slot.player} open pick`} />
              <span className="slot-copy">
                <strong>{slot.player}</strong>
                <span>{champion?.name ?? slot.role}</span>
                {isActive && <em>Picking...</em>}
              </span>
              <span className="swap-icon" aria-hidden="true">
                Role
              </span>
            </button>
          )
        })}
      </div>
      <div className="ban-strip">
        <strong>Bans</strong>
        <div>
          {bans.map((banId, index) => {
            const champion = championById(banId)
            const isActive = team === activeTeam && index === activeBanIndex
            const isLocked =
              feedback?.action === 'ban' &&
              feedback.team === team &&
              feedback.banIndex === index
            return (
              <span className={`ban-chip ${isActive ? 'active' : ''} ${isLocked ? 'locked' : ''}`} key={`${team}-ban-${index}`}>
                <ChampionPortrait id={banId} alt={champion?.name ?? 'Empty ban'} />
                {banId && <i aria-hidden="true">x</i>}
              </span>
            )
          })}
        </div>
      </div>
    </aside>
  )
}

function App() {
  const [blueSlots, setBlueSlots] = useState(initialBlueSlots)
  const [redSlots, setRedSlots] = useState(initialRedSlots)
  const [blueBans, setBlueBans] = useState(emptyBans)
  const [redBans, setRedBans] = useState(emptyBans)
  const [stepIndex, setStepIndex] = useState(0)
  const [pendingChampionId, setPendingChampionId] = useState<string>()
  const [feedback, setFeedback] = useState<LockFeedback | null>(null)
  const [search, setSearch] = useState('')
  const [roleFilter, setRoleFilter] = useState<Role | 'ALL'>('ALL')
  const [sortKey, setSortKey] = useState<SortKey>('winRate')

  const currentStep = draftSteps[stepIndex]
  const isComplete = !currentStep
  const activeTeam = currentStep?.team ?? 'blue'
  const activeRole = currentPickRole(currentStep, blueSlots, redSlots)
  const focusRole = activeRole ?? nextPickRole(stepIndex, blueSlots, redSlots)
  const activeBanIndex = currentStep?.action === 'ban' ? currentStep.banIndex : undefined
  const activeSlotIndex = currentStep?.action === 'pick' ? currentStep.slotIndex : undefined
  const pendingChampion = championById(pendingChampionId)

  const [prediction, setPrediction] = useState<Prediction>(() =>
    localPrediction({
      bluePicks: initialBlueSlots,
      redPicks: initialRedSlots,
      blueBans: emptyBans,
      redBans: emptyBans,
      activeTeam: 'blue',
      activeAction: 'ban',
      stepIndex: 0,
    }),
  )

  const draftPayload = useMemo<DraftPayload>(
    () => ({
      bluePicks: blueSlots,
      redPicks: redSlots,
      blueBans,
      redBans,
      activeTeam,
      activeAction: currentStep?.action ?? 'pick',
      activeRole,
      stepIndex,
    }),
    [activeRole, activeTeam, blueBans, blueSlots, currentStep?.action, redBans, redSlots, stepIndex],
  )

  useEffect(() => {
    let cancelled = false

    requestPrediction(draftPayload).then((nextPrediction) => {
      if (!cancelled) setPrediction(nextPrediction)
    })

    return () => {
      cancelled = true
    }
  }, [draftPayload])

  const selectedIds = useMemo(() => {
    return new Set([
      ...blueSlots.map((slot) => slot.championId),
      ...redSlots.map((slot) => slot.championId),
      ...blueBans,
      ...redBans,
    ].filter(Boolean))
  }, [blueBans, blueSlots, redBans, redSlots])

  const visibleChampions = useMemo(() => {
    const query = search.trim().toLowerCase()

    return champions
      .filter((champion) => !selectedIds.has(champion.id))
      .filter((champion) => roleFilter === 'ALL' || rolesForChampion(champion).includes(roleFilter))
      .filter((champion) => {
        if (!query) return true
        return (
          champion.name.toLowerCase().includes(query) ||
          champion.title.toLowerCase().includes(query) ||
          champion.tags.join(' ').toLowerCase().includes(query)
        )
      })
      .sort((a, b) => {
        const aMetrics = championMetrics(a, focusRole)
        const bMetrics = championMetrics(b, focusRole)
        if (sortKey === 'name') return a.name.localeCompare(b.name)
        if (sortKey === 'synergy') return bMetrics.synergy - aMetrics.synergy
        if (sortKey === 'counter') return bMetrics.counter - aMetrics.counter
        return bMetrics.winRate - aMetrics.winRate
      })
  }, [focusRole, roleFilter, search, selectedIds, sortKey])

  const statRows = useMemo(() => {
    return visibleChampions.slice(0, 5).map((champion) => ({
      champion,
      metrics: championMetrics(champion, focusRole),
    }))
  }, [focusRole, visibleChampions])

  const recommended = statRows.slice(0, 3)

  function selectChampion(championId: string) {
    if (!currentStep) return
    setPendingChampionId(championId)
  }

  function confirmSelection() {
    if (!currentStep || !pendingChampionId || !pendingChampion) return

    if (currentStep.action === 'ban' && currentStep.banIndex !== undefined) {
      const updateBans = (current: string[]) =>
        current.map((banId, index) => (index === currentStep.banIndex ? pendingChampionId : banId))

      if (currentStep.team === 'blue') {
        setBlueBans(updateBans)
      } else {
        setRedBans(updateBans)
      }
    }

    if (currentStep.action === 'pick' && currentStep.slotIndex !== undefined) {
      const updateSlots = (current: Slot[]) =>
        current.map((slot, index) =>
          index === currentStep.slotIndex ? { ...slot, championId: pendingChampionId } : slot,
        )

      if (currentStep.team === 'blue') {
        setBlueSlots(updateSlots)
      } else {
        setRedSlots(updateSlots)
      }
    }

    const stamp = window.Date.now()
    setFeedback({
      action: currentStep.action,
      banIndex: currentStep.banIndex,
      championName: pendingChampion.name,
      slotIndex: currentStep.slotIndex,
      stamp,
      team: currentStep.team,
    })
    window.setTimeout(() => {
      setFeedback((current) => (current?.stamp === stamp ? null : current))
    }, 1100)
    setPendingChampionId(undefined)
    setStepIndex((current) => Math.min(current + 1, draftSteps.length))
  }

  return (
    <main className="draft-shell">
      <header className="top-bar">
        <div className="brand-lockup">
          <div className="brand-crest" aria-hidden="true">R</div>
          <strong>Rift Draft</strong>
        </div>
        <nav className="main-tabs" aria-label="Draft mode">
          <button className="selected" type="button">Ranked Draft</button>
        </nav>
        <div className={`phase-card ${activeTeam}`}>
          <span>{currentStep?.phase ?? 'Draft Complete'}</span>
          <em>
            {currentStep
              ? `${teamName(currentStep.team)} is ${currentStep.action === 'ban' ? 'banning' : 'picking'}`
              : 'All bans and picks are locked'}
          </em>
        </div>
        <div className="header-tools" aria-label="Settings">
          <button type="button" aria-label="Settings">&#9881;</button>
        </div>
      </header>

      <section className="scoreboard" aria-label="Projected win chance">
        <div className="team-score blue">
          <TeamMark team="blue" />
          <div>
            <span>Blue Team</span>
            <strong>{prediction.blueWinRate.toFixed(1)}%</strong>
            <em>Projected Win Chance</em>
          </div>
        </div>
        <div className="team-score red">
          <TeamMark team="red" />
          <div>
            <span>Red Team</span>
            <strong>{prediction.redWinRate.toFixed(1)}%</strong>
            <em>Projected Win Chance</em>
          </div>
        </div>
      </section>

      <section className="draft-grid">
        <TeamPanel
          activeBanIndex={activeBanIndex}
          activeSlotIndex={activeSlotIndex}
          activeTeam={activeTeam}
          bans={blueBans}
          feedback={feedback}
          slots={blueSlots}
          team="blue"
        />

        <section className="champion-board" aria-label="Champion selection">
          <div className="toolbar">
            <label className="search-box">
              <span aria-hidden="true">Search</span>
              <input
                aria-label="Search champions"
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search champions..."
                type="search"
                value={search}
              />
            </label>
            <div className="role-filters" aria-label="Role filter">
              <button className={roleFilter === 'ALL' ? 'selected' : ''} onClick={() => setRoleFilter('ALL')} type="button">
                All
              </button>
              {roles.map((role) => (
                <button
                  className={roleFilter === role ? 'selected' : ''}
                  key={role}
                  onClick={() => setRoleFilter(role)}
                  type="button"
                >
                  {role}
                </button>
              ))}
            </div>
            <label className="sort-control">
              <span>Sort by:</span>
              <select onChange={(event) => setSortKey(event.target.value as SortKey)} value={sortKey}>
                <option value="winRate">Win Rate</option>
                <option value="synergy">Synergy</option>
                <option value="counter">Counter</option>
                <option value="name">Name</option>
              </select>
            </label>
          </div>

          <div className={`action-banner ${activeTeam}`}>
            <div className="action-copy">
              <strong>{currentStep?.label ?? 'Draft complete'}</strong>
              <span>
                {isComplete
                  ? 'Champion selection is locked.'
                  : `Select a champion, then confirm the ${currentStep.action}.`}
              </span>
            </div>
            <div className="pending-selection">
              <ChampionPortrait id={pendingChampionId} alt={pendingChampion?.name ?? 'No selected champion'} />
              <span>
                <strong>{pendingChampion?.name ?? 'No champion selected'}</strong>
                <em>{currentStep ? `${teamName(currentStep.team)} ${currentStep.action}` : 'Draft complete'}</em>
              </span>
              <button disabled={!pendingChampionId || isComplete} onClick={confirmSelection} type="button">
                Confirm {currentStep?.action ?? 'pick'}
              </button>
            </div>
          </div>

          {feedback && (
            <div className={`lock-feedback ${feedback.team}`} key={feedback.stamp}>
              <strong>{feedback.championName}</strong>
              <span>{teamName(feedback.team)} {feedback.action === 'ban' ? 'ban locked' : 'pick locked'}</span>
            </div>
          )}

          <div className="champion-grid">
            {visibleChampions.length === 0 && (
              <div className="champion-card placeholder">No matches</div>
            )}
            {visibleChampions.map((champion, index) => {
              const metrics = championMetrics(champion, focusRole)

              return (
                <button
                  className={`champion-card ${pendingChampionId === champion.id ? 'selected' : ''}`}
                  disabled={isComplete}
                  key={champion.id}
                  onClick={() => selectChampion(champion.id)}
                  type="button"
                >
                  {(index === 0 || index === 3 || index === 5) && <span className="favorite">*</span>}
                  <ChampionPortrait id={champion.id} alt={champion.name} />
                  <span className="champion-card-copy">
                    <strong>{champion.name}</strong>
                    <span>
                      {metrics.winRate.toFixed(1)}% <em>{metrics.synergy}</em>
                    </span>
                  </span>
                  <i aria-hidden="true">{currentStep?.action === 'ban' ? 'Ban' : 'Pick'}</i>
                </button>
              )
            })}
          </div>
        </section>

        <TeamPanel
          activeBanIndex={activeBanIndex}
          activeSlotIndex={activeSlotIndex}
          activeTeam={activeTeam}
          bans={redBans}
          feedback={feedback}
          slots={redSlots}
          team="red"
        />
      </section>

      <section className="lower-grid">
        <section className="data-panel stats-panel">
          <div className="panel-tabs">
            <button className="selected" type="button">Champion Stats</button>
            <button type="button">Matchups</button>
            <button type="button">Synergy</button>
            <button type="button">Counters</button>
          </div>
          <div className="stats-table" role="table" aria-label="Champion stats">
            <div className="stats-row heading" role="row">
              <span>Champion</span>
              <span>Role</span>
              <span>Win Rate</span>
              <span>Synergy</span>
              <span>Counter</span>
              <span>Sample Size</span>
            </div>
            {statRows.map(({ champion, metrics }) => (
              <div className="stats-row" key={champion.id} role="row">
                <span className="stat-champion">
                  <ChampionPortrait id={champion.id} alt={champion.name} />
                  {champion.name}
                </span>
                <span>{focusRole.charAt(0) + focusRole.slice(1).toLowerCase()}</span>
                <strong>{metrics.winRate.toFixed(1)}%</strong>
                <span>{metrics.synergy}</span>
                <span>{metrics.counter}</span>
                <span>{metrics.sampleSize.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="data-panel recommendation-panel">
          <div className="assistant-header">
            <h2>Draft Assistant</h2>
          </div>
          <div className="recommendation-grid">
            {recommended.map(({ champion, metrics }, index) => (
              <button
                className={`recommendation-card ${pendingChampionId === champion.id ? 'selected' : ''}`}
                disabled={isComplete}
                key={champion.id}
                onClick={() => selectChampion(champion.id)}
                type="button"
              >
                <span className="rank">{index + 1}</span>
                <ChampionPortrait id={champion.id} alt={champion.name} />
                <div>
                  <strong>{champion.name}</strong>
                  <em>{metrics.winRate.toFixed(1)}% winrate</em>
                </div>
              </button>
            ))}
          </div>
        </section>
      </section>

      <div className="backend-entry" aria-live="polite">
        Model source: {prediction.source === 'backend' ? 'Backend API' : 'Local fallback'}
      </div>
    </main>
  )
}

export default App

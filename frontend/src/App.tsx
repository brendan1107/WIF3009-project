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
  const blueFilled = payload.bluePicks.filter((slot) => slot.championId).length
  const redFilled = payload.redPicks.filter((slot) => slot.championId).length

  // Gate: return 50/50 when either team has no picks yet.
  // Without an opposing draft there is no signal to compare against.
  if (blueFilled === 0 || redFilled === 0) {
    return {
      blueWinRate: 50.0,
      redWinRate: 50.0,
      source: 'local',
    }
  }

  const blueScore = teamScore(payload.bluePicks)
  const redScore = teamScore(payload.redPicks)
  const blueBans = payload.blueBans.filter(Boolean).length
  const redBans = payload.redBans.filter(Boolean).length
  const banPressure = (redBans - blueBans) * 0.18
  const activeBonus = payload.activeTeam === 'blue' ? 0.35 : -0.35
  const rawBlueWinRate = clamp(
    50 + (blueScore - redScore) * 0.16 + (blueFilled - redFilled) * 0.72 + banPressure + activeBonus,
    38,
    62,
  )

  // Linear completeness scaling: shrink deviation from 50% proportionally
  // to how complete the least-filled side is (mirrors backend behaviour).
  const completenessFactor = Math.min(blueFilled, redFilled) / 5
  const blueWinRate = 50 + (rawBlueWinRate - 50) * completenessFactor

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
  activeSlotIndex,
  activeTeam,
  feedback,
  slots,
  team,
}: {
  activeSlotIndex?: number
  activeTeam: Team
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
    </aside>
  )
}

function App() {
  const [scale, setScale] = useState(1)
  const [assistantOpen, setAssistantOpen] = useState(false)

  useEffect(() => {
    const handleResize = () => {
      const baseWidth = 1920
      const baseHeight = 1080
      setScale(Math.min(window.innerWidth / baseWidth, window.innerHeight / baseHeight))
    }
    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const [activeTab, setActiveTab] = useState<'stats' | 'metaProver'>('stats')
  const [opChampions, setOpChampions] = useState<Array<{
    champion: string
    mean_contribution: number
    mean_abs_contribution: number
    count: number
  }>>([])
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

  const [draftWarning, setDraftWarning] = useState<string>("")
  const [recommendations, setRecommendations] = useState<string>("")
  const [counterAnalysis, setCounterAnalysis] = useState<string>("")
  const [loadingCoach, setLoadingCoach] = useState<boolean>(false)
  const [champMetrics, setChampMetrics] = useState<{
    winrates: Record<string, number>
    synergies: Record<string, number>
    globalAvgWr: number
  } | null>(null)
  const [shapDrivers, setShapDrivers] = useState<Array<{
    feature: string
    value: number
    impact_on_win_prob: number
  }>>([])

  const draftPayload = useMemo<DraftPayload>(
    () => {
      let nextBluePicks = blueSlots
      let nextRedPicks = redSlots

      if (pendingChampionId && currentStep?.action === 'pick' && activeSlotIndex !== undefined) {
        const updateSlots = (slotsList: Slot[]) =>
          slotsList.map((slot, index) =>
            index === activeSlotIndex ? { ...slot, championId: pendingChampionId } : slot
          )
        if (activeTeam === 'blue') {
          nextBluePicks = updateSlots(blueSlots)
        } else {
          nextRedPicks = updateSlots(redSlots)
        }
      }

      return {
        bluePicks: nextBluePicks,
        redPicks: nextRedPicks,
        blueBans,
        redBans,
        activeTeam,
        activeAction: currentStep?.action ?? 'pick',
        activeRole,
        stepIndex,
      }
    },
    [activeRole, activeTeam, blueBans, blueSlots, currentStep?.action, redBans, redSlots, stepIndex, pendingChampionId, activeSlotIndex],
  )

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const apiBase = import.meta.env.VITE_WINRATE_API_URL
          ? import.meta.env.VITE_WINRATE_API_URL.replace("/predict", "")
          : "http://localhost:8000/api/v1"
        const response = await fetch(`${apiBase}/champions/metrics`)
        if (response.ok) {
          const data = await response.json()
          setChampMetrics({
            winrates: data.winrates,
            synergies: data.synergies,
            globalAvgWr: data.global_avg_wr,
          })
        }

        const opResponse = await fetch(`${apiBase}/champions/op`)
        if (opResponse.ok) {
          const opData = await opResponse.json()
          setOpChampions(opData.op_champions || [])
        }
      } catch (err) {
        console.error("Failed to fetch champion metrics from backend:", err)
      }
    }
    fetchMetrics()
  }, [])

  const getRealMetrics = useMemo(() => {
    return (champion: Champion, role: Role) => {
      const local = championMetrics(champion, role)
      if (!champMetrics) return local

      const name = champion.name
      const realWr = champMetrics.winrates[name] !== undefined
        ? champMetrics.winrates[name] * 100
        : champMetrics.globalAvgWr * 100

      const allySlots = activeTeam === 'blue' ? blueSlots : redSlots
      const allyNames = allySlots
        .map(s => championById(s.championId)?.name)
        .filter((n): n is string => !!n && n !== name)

      let synergySum = 0
      let synergyCount = 0
      for (const allyName of allyNames) {
        const key = [name, allyName].sort().join('_')
        const pairWr = champMetrics.synergies[key]
        if (pairWr !== undefined) {
          synergySum += pairWr
          synergyCount++
        }
      }
      const synergy = synergyCount > 0
        ? Math.round((synergySum / synergyCount) * 100)
        : local.synergy

      const enemySlots = activeTeam === 'blue' ? redSlots : blueSlots
      const enemyNames = enemySlots
        .map(s => championById(s.championId)?.name)
        .filter((n): n is string => !!n)

      let enemyWrSum = 0
      let enemyCount = 0
      for (const enemyName of enemyNames) {
        const enemyWr = champMetrics.winrates[enemyName]
        if (enemyWr !== undefined) {
          enemyWrSum += enemyWr
          enemyCount++
        }
      }
      const counter = enemyCount > 0
        ? Math.round(clamp(50 + (realWr / 100 - enemyWrSum / enemyCount) * 100, 30, 95))
        : local.counter

      return {
        winRate: realWr,
        synergy,
        counter,
        sampleSize: local.sampleSize
      }
    }
  }, [champMetrics, activeTeam, blueSlots, redSlots])

  useEffect(() => {
    let cancelled = false

    const isPickStep = currentStep?.action === 'pick'
    const hasPending = !!pendingChampionId

    if (isPickStep || !hasPending) {
      requestPrediction(draftPayload).then((nextPrediction) => {
        if (!cancelled) setPrediction(nextPrediction)
      })
    }

    const shouldCallAdvisor = !hasPending

    if (shouldCallAdvisor) {
      const fetchAdvice = async () => {
        setLoadingCoach(true)
        try {
          const response = await fetch(import.meta.env.VITE_COACHING_API_URL || "http://localhost:8000/api/v1/agent", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              draft_state: draftPayload
            })
          })

          if (!response.ok) throw new Error("Coaching API error")

          const result = await response.json()
          if (!cancelled) {
            setDraftWarning(result.draft_warning || "")
            setRecommendations(result.recommendations || "")
            setCounterAnalysis(result.counter_analysis || "")
            if (result.top_drivers) {
              setShapDrivers(result.top_drivers)
            }
          }
        } catch (err) {
          console.error("Co-pilot feedback failed:", err)
          if (!cancelled) {
            setDraftWarning("Composition threat analyzer offline.")
            setRecommendations("Check server connections.")
            setCounterAnalysis("Matchup counter analysis offline.")
          }
        } finally {
          if (!cancelled) setLoadingCoach(false)
        }
      }

      fetchAdvice()
    }

    return () => {
      cancelled = true
    }
  }, [draftPayload, pendingChampionId, currentStep?.action])

  const selectedIds = useMemo(() => {
    return new Set([
      ...blueSlots.map((slot) => slot.championId),
      ...redSlots.map((slot) => slot.championId),
      ...blueBans,
      ...redBans,
    ].filter(Boolean))
  }, [blueBans, blueSlots, redBans, redSlots])

  const recommendedChampionNames = useMemo<string[]>(() => {
    if (!recommendations) return []
    const names: string[] = []
    recommendations.split(/(?=\d+\.\s+)/).forEach((item) => {
      const trimmed = item.trim()
      const match = trimmed.match(/^(\d+\.\s+)([^:]+):(.*)$/)
      if (match) {
        names.push(match[2].trim())
      }
    })
    return names
  }, [recommendations])

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
        const aRec = recommendedChampionNames.includes(a.name)
        const bRec = recommendedChampionNames.includes(b.name)
        if (aRec && !bRec) return -1
        if (!aRec && bRec) return 1

        const aMetrics = getRealMetrics(a, focusRole)
        const bMetrics = getRealMetrics(b, focusRole)
        if (sortKey === 'name') return a.name.localeCompare(b.name)
        if (sortKey === 'synergy') return bMetrics.synergy - aMetrics.synergy
        if (sortKey === 'counter') return bMetrics.counter - aMetrics.counter
        return bMetrics.winRate - aMetrics.winRate
      })
  }, [focusRole, roleFilter, search, selectedIds, sortKey, getRealMetrics, recommendedChampionNames])

  const statRows = useMemo(() => {
    return visibleChampions.slice(0, 5).map((champion) => ({
      champion,
      metrics: getRealMetrics(champion, focusRole),
    }))
  }, [focusRole, visibleChampions, getRealMetrics])

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
    <div className="scaler-wrapper">
      <main className="draft-shell" style={{ transform: `translate(-50%, -50%) scale(${scale})` }}>
        <div className="left-layout">
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
              <div className="scoreboard-bans blue">
                {blueBans.map((banId, index) => {
                  const champion = championById(banId)
                  const isActive = activeTeam === 'blue' && index === activeBanIndex
                  const isLocked =
                    feedback?.action === 'ban' &&
                    feedback.team === 'blue' &&
                    feedback.banIndex === index
                  return (
                    <span className={`ban-chip ${isActive ? 'active' : ''} ${isLocked ? 'locked' : ''}`} key={`blue-ban-${index}`}>
                      <ChampionPortrait id={banId} alt={champion?.name ?? 'Empty ban'} />
                      {banId && <i aria-hidden="true">x</i>}
                    </span>
                  )
                })}
              </div>
            </div>
            <div className="team-score red">
              <div className="scoreboard-bans red">
                {redBans.map((banId, index) => {
                  const champion = championById(banId)
                  const isActive = activeTeam === 'red' && index === activeBanIndex
                  const isLocked =
                    feedback?.action === 'ban' &&
                    feedback.team === 'red' &&
                    feedback.banIndex === index
                  return (
                    <span className={`ban-chip ${isActive ? 'active' : ''} ${isLocked ? 'locked' : ''}`} key={`red-ban-${index}`}>
                      <ChampionPortrait id={banId} alt={champion?.name ?? 'Empty ban'} />
                      {banId && <i aria-hidden="true">x</i>}
                    </span>
                  )
                })}
              </div>
              <div>
                <span>Red Team</span>
                <strong>{prediction.redWinRate.toFixed(1)}%</strong>
                <em>Projected Win Chance</em>
              </div>
              <TeamMark team="red" />
            </div>
          </section>

          <section className="draft-grid">
            <TeamPanel
              activeSlotIndex={activeSlotIndex}
              activeTeam={activeTeam}
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
                  const metrics = getRealMetrics(champion, focusRole)
                  const isRec = recommendedChampionNames.includes(champion.name)

                  return (
                    <button
                      className={`champion-card ${pendingChampionId === champion.id ? 'selected' : ''} ${isRec ? 'recommended' : ''}`}
                      disabled={isComplete}
                      key={champion.id}
                      onClick={() => selectChampion(champion.id)}
                      type="button"
                    >
                      {isRec ? (
                        <span className="favorite" style={{ color: '#ffd700' }}>★</span>
                      ) : (
                        (index === 0 || index === 3 || index === 5) && <span className="favorite">*</span>
                      )}
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
              activeSlotIndex={activeSlotIndex}
              activeTeam={activeTeam}
              feedback={feedback}
              slots={redSlots}
              team="red"
            />
          </section>

          <section className="data-panel stats-panel">
            <div className="panel-tabs" style={{ gridTemplateColumns: "repeat(2, minmax(0, 1fr))" }}>
              <button
                className={activeTab === 'stats' ? 'selected' : ''}
                onClick={() => setActiveTab('stats')}
                type="button"
              >
                Champion Stats
              </button>
              <button
                className={activeTab === 'metaProver' ? 'selected' : ''}
                onClick={() => setActiveTab('metaProver')}
                type="button"
              >
                Meta-Prover (OP)
              </button>
            </div>
            {activeTab === 'stats' && (
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
            )}
            {activeTab === 'metaProver' && (
              <div className="stats-table" role="table" aria-label="Meta-Prover OP champions" style={{ gridTemplateRows: "repeat(6, minmax(0, 1fr))" }}>
                <div className="stats-row heading" role="row" style={{ gridTemplateColumns: "minmax(150px, 1.3fr) 1fr 1fr 0.9fr" }}>
                  <span>Champion</span>
                  <span>Mean SHAP Impact</span>
                  <span>Mean Abs SHAP</span>
                  <span>Games Analyzed</span>
                </div>
                {opChampions.slice(0, 5).map((op) => {
                  const champObj = champions.find(c => c.name === op.champion)
                  const isPositive = op.mean_contribution >= 0
                  return (
                    <div className="stats-row" key={op.champion} role="row" style={{ gridTemplateColumns: "minmax(150px, 1.3fr) 1fr 1fr 0.9fr" }}>
                      <span className="stat-champion">
                        <ChampionPortrait id={champObj?.id} alt={op.champion} />
                        {op.champion}
                      </span>
                      <strong className={isPositive ? 'positive' : 'negative'} style={{ color: isPositive ? '#6fca68' : '#ff4c4c' }}>
                        {isPositive ? '+' : ''}{(op.mean_contribution * 100).toFixed(2)}%
                      </strong>
                      <span>{(op.mean_abs_contribution * 100).toFixed(2)}%</span>
                      <span>{op.count}</span>
                    </div>
                  )
                })}
              </div>
            )}
          </section>
        </div>

        <aside className={`data-panel recommendation-panel ${assistantOpen ? 'expanded' : ''}`}>
          <button
            aria-expanded={assistantOpen}
            aria-label={`${assistantOpen ? 'Collapse' : 'Expand'} draft assistant`}
            className="assistant-rail"
            onClick={() => setAssistantOpen((isOpen) => !isOpen)}
            type="button"
          >
            <span className="assistant-rail-mark">AI</span>
            <span className="assistant-rail-label">Coach</span>
            <span className={`assistant-rail-dot ${loadingCoach ? 'loading' : ''}`} />
          </button>

          <div className="assistant-drawer-content">
            <div className="assistant-header">
              <h2>Draft Assistant</h2>
              <button
                aria-label="Collapse draft assistant"
                className="assistant-close"
                onClick={() => setAssistantOpen(false)}
                type="button"
              >
                x
              </button>
            </div>

            <div className="co-pilot-dashboard">
              <div className="co-pilot-review">
                <h3>Tactical Review</h3>
                {loadingCoach ? (
                  <div className="co-pilot-loading">
                    <span className="pulse-dot"></span>
                    <span>Calculating SHAP contributions...</span>
                  </div>
                ) : (
                  <div className="co-pilot-content">
                    {draftWarning || counterAnalysis || recommendations ? (
                      <div className="co-pilot-sections">
                        {(draftWarning || counterAnalysis) && (
                          <div className="co-pilot-left-col">
                            {draftWarning && (
                              <div className="co-pilot-section warning-card">
                                <h4>Composition Warning</h4>
                                <p>{draftWarning}</p>
                              </div>
                            )}
                            {counterAnalysis && (
                              <div className="co-pilot-section counter-card">
                                <h4>Tactical Counters</h4>
                                <p>{counterAnalysis}</p>
                              </div>
                            )}
                          </div>
                        )}
                        {recommendations && (
                          <div className="co-pilot-section recommendations-card">
                            <h4>Strategic Recommendations</h4>
                            <div className="recommendations-list">
                              {recommendations.split(/(?=\d+\.\s+)/).map((item, idx) => {
                                const trimmed = item.trim()
                                if (!trimmed) return null
                                const match = trimmed.match(/^(\d+\.\s+)([^:]+):(.*)$/)
                                if (match) {
                                  const [, numberPrefix, champName, rest] = match
                                  return (
                                    <div key={idx} className="recommendation-item">
                                      <span className="recommendation-num-champ">
                                        <strong>{numberPrefix}</strong>
                                        <strong>{champName}</strong>:
                                      </span>
                                      <span>{rest}</span>
                                    </div>
                                  )
                                }
                                return (
                                  <div key={idx} className="recommendation-item fallback">
                                    {trimmed}
                                  </div>
                                )
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p>Pick a champion or lock a ban to trigger tactical co-pilot advice.</p>
                    )}
                  </div>
                )}
              </div>

              <div className="shap-drivers-panel">
                <h3>Tactical Winrate Drivers</h3>
                <div className="shap-drivers-list">
                  {shapDrivers.length === 0 ? (
                    <div className="co-pilot-loading">No active drivers</div>
                  ) : (
                    shapDrivers.map((driver) => {
                      const isPositive = driver.impact_on_win_prob >= 0
                      const percentVal = Math.abs(driver.impact_on_win_prob) * 100
                      const barWidth = clamp(percentVal * 6, 5, 100)
                      const displayVal = `${isPositive ? '+' : '-'}${percentVal.toFixed(1)}%`

                      // Driver labels map
                      const featureLabels: Record<string, string> = {
                        blue_synergy: 'Blue Synergy',
                        red_synergy: 'Red Synergy',
                        blue_team_avg_wr: 'Blue Avg WR',
                        red_team_avg_wr: 'Red Avg WR',
                        wr_diff: 'Winrate Diff',
                        synergy_diff: 'Synergy Diff',
                        blue_top_wr: 'Blue Top WR',
                        blue_jng_wr: 'Blue Jungle WR',
                        blue_mid_wr: 'Blue Mid WR',
                        blue_bot_wr: 'Blue Bot WR',
                        blue_sup_wr: 'Blue Support WR',
                        red_top_wr: 'Red Top WR',
                        red_jng_wr: 'Red Jungle WR',
                        red_mid_wr: 'Red Mid WR',
                        red_bot_wr: 'Red Bot WR',
                        red_sup_wr: 'Red Support WR',
                        blue_top_enc: 'Blue Top Pick',
                        blue_jng_enc: 'Blue Jungle Pick',
                        blue_mid_enc: 'Blue Mid Pick',
                        blue_bot_enc: 'Blue Bot Pick',
                        blue_sup_enc: 'Blue Support Pick',
                        red_top_enc: 'Red Top Pick',
                        red_jng_enc: 'Red Jungle Pick',
                        red_mid_enc: 'Red Mid Pick',
                        red_bot_enc: 'Red Bot Pick',
                        red_sup_enc: 'Red Support Pick',
                      }

                      const formatFeatureName = (feat: string) => {
                        if (featureLabels[feat]) return featureLabels[feat]
                        return feat
                          .replace(/_enc$/, ' Pick')
                          .replace(/_wr$/, ' WR')
                          .split('_')
                          .map(w => w.charAt(0).toUpperCase() + w.slice(1))
                          .join(' ')
                      }

                      return (
                        <div className="shap-driver-row" key={driver.feature}>
                          <span className="shap-driver-label" title={driver.feature}>
                            {formatFeatureName(driver.feature)}
                          </span>
                          <div className="driver-bar-bg">
                            <div
                              className={`driver-bar-fill ${isPositive ? 'positive' : 'negative'}`}
                              style={{ width: `${barWidth}%` }}
                            />
                          </div>
                          <span className={`driver-bar-value ${isPositive ? 'positive' : 'negative'}`}>
                            {displayVal}
                          </span>
                        </div>
                      )
                    })
                  )}
                </div>
              </div>
            </div>
          </div>
        </aside>
        <div className="backend-entry" aria-live="polite">
          Model source: {prediction.source === 'backend' ? 'Backend API' : 'Local fallback'}
        </div>
      </main>
    </div>
  )
}

export default App

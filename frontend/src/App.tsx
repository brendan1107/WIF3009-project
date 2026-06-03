import { useEffect, useMemo, useRef, useState } from 'react'
import championData from './assets/data/champions.json'
import './App.css'

type Role = 'TOP' | 'JUNGLE' | 'MID' | 'BOTTOM' | 'SUPPORT'
type Team = 'blue' | 'red'
type SortKey = 'winRate' | 'name'

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
  player: string
  championId?: string
  role?: Role
}

type DraftStep = {
  phase: 'Pick Phase 1' | 'Pick Phase 2'
  action: 'pick'
  team: Team
  label: string
  slotIndex?: number
}

type DraftPayload = {
  bluePicks: Slot[]
  redPicks: Slot[]
  activeTeam: Team
  activeAction: 'pick'
  activeRole?: Role
  stepIndex: number
}

type Prediction = {
  blueWinRate: number
  redWinRate: number
  source: 'backend' | 'local'
}

type LockFeedback = {
  championName: string
  role?: Role
  slotIndex?: number
  stamp: number
  team: Team
}

type ChampionOption = {
  champion: Champion
  entityId: string
  role: Role
}

type ReplacementTarget = {
  team: Team
  slotIndex: number
  role: Role
}

type ApiSlot = {
  role: Role
  player: string
  championId: string
}

type ApiDraftPayload = {
  bluePicks: ApiSlot[]
  redPicks: ApiSlot[]
  blueBans: string[]
  redBans: string[]
  activeTeam: Team
  activeAction: 'pick'
  activeRole?: Role
  stepIndex: number
}

type ApiCandidateOption = {
  entityId: string
  championId: string
  championName: string
  role: Role
}

type PredictOptionsResponse = {
  optionWinRates: Record<string, number>
}

const championIcons = import.meta.glob('./assets/champion/*.png', {
  eager: true,
  import: 'default',
  query: '?url',
}) as Record<string, string>

const roles: Role[] = ['TOP', 'JUNGLE', 'MID', 'BOTTOM', 'SUPPORT']

const draftSteps: DraftStep[] = [
  { phase: 'Pick Phase 1', action: 'pick', team: 'blue', slotIndex: 0, label: 'Blue Pick 1' },
  { phase: 'Pick Phase 1', action: 'pick', team: 'red', slotIndex: 0, label: 'Red Pick 1' },
  { phase: 'Pick Phase 1', action: 'pick', team: 'red', slotIndex: 1, label: 'Red Pick 2' },
  { phase: 'Pick Phase 1', action: 'pick', team: 'blue', slotIndex: 1, label: 'Blue Pick 2' },
  { phase: 'Pick Phase 1', action: 'pick', team: 'blue', slotIndex: 2, label: 'Blue Pick 3' },
  { phase: 'Pick Phase 1', action: 'pick', team: 'red', slotIndex: 2, label: 'Red Pick 3' },
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
  Ashe: ['BOTTOM', 'SUPPORT'],
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
  { player: 'Player 1' },
  { player: 'Player 2' },
  { player: 'Player 3' },
  { player: 'Player 4' },
  { player: 'Player 5' },
]

const initialRedSlots: Slot[] = [
  { player: 'Enemy 1' },
  { player: 'Enemy 2' },
  { player: 'Enemy 3' },
  { player: 'Enemy 4' },
  { player: 'Enemy 5' },
]

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
  const sampleSize = 8700 + (Number(champion.key) % 8500)

  return { winRate, sampleSize }
}

function teamScore(slots: Slot[]) {
  return slots.reduce((score, slot) => {
    const champion = championById(slot.championId)
    if (!champion || !slot.role) return score
    const metrics = championMetrics(champion, slot.role)
    return score + metrics.winRate
  }, 0)
}

function localPrediction(payload: DraftPayload): Prediction {
  const blueFilled = payload.bluePicks.filter((slot) => slot.championId).length
  const redFilled = payload.redPicks.filter((slot) => slot.championId).length
  if (blueFilled === 0 && redFilled === 0) {
    return {
      blueWinRate: 50.0,
      redWinRate: 50.0,
      source: 'local',
    }
  }

  const blueScore = teamScore(payload.bluePicks)
  const redScore = teamScore(payload.redPicks)
  const activeBonus = payload.activeTeam === 'blue' ? 0.35 : -0.35
  const blueWinRate = clamp(
    50 + (blueScore - redScore) * 0.16 + (blueFilled - redFilled) * 0.72 + activeBonus,
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
    const apiPayload = toApiDraftPayload(payload)
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(apiPayload),
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

function optionId(championId: string, role: Role) {
  return `${championId}-${role}`
}

function roleLabel(role?: Role) {
  return role ? role.charAt(0) + role.slice(1).toLowerCase() : 'Role pending'
}

function apiSlots(slots: Slot[]) {
  return slots.flatMap((slot): ApiSlot[] => {
    const champion = championById(slot.championId)
    if (!champion || !slot.role) return []
    return [
      {
        role: slot.role,
        player: slot.player,
        championId: champion.name,
      },
    ]
  })
}

function toApiDraftPayload(payload: DraftPayload): ApiDraftPayload {
  return {
    bluePicks: apiSlots(payload.bluePicks),
    redPicks: apiSlots(payload.redPicks),
    blueBans: [],
    redBans: [],
    activeTeam: payload.activeTeam,
    activeAction: payload.activeAction,
    activeRole: payload.activeRole,
    stepIndex: payload.stepIndex,
  }
}

function predictOptionsEndpoint() {
  const endpoint = import.meta.env.VITE_WINRATE_API_URL
  if (!endpoint) return undefined
  const normalizedEndpoint = endpoint.replace(/\/$/, '')
  return normalizedEndpoint.endsWith('/predict')
    ? normalizedEndpoint.replace(/\/predict$/, '/predict-options')
    : `${normalizedEndpoint}/predict-options`
}

function toApiCandidateOptions(options: ChampionOption[]): ApiCandidateOption[] {
  return options.map(({ champion, entityId, role }) => ({
    entityId,
    championId: champion.id,
    championName: champion.name,
    role,
  }))
}

async function requestOptionWinRates(draftState: DraftPayload, options: ChampionOption[]) {
  const endpoint = predictOptionsEndpoint()
  if (!endpoint || options.length === 0) return undefined

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      draftState: toApiDraftPayload(draftState),
      options: toApiCandidateOptions(options),
    }),
  })

  if (!response.ok) {
    throw new Error(`Predict options API returned ${response.status}`)
  }

  return (await response.json()) as PredictOptionsResponse
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
  canReplace,
  feedback,
  onReplaceSlot,
  slots,
  team,
}: {
  activeSlotIndex?: number
  activeTeam: Team
  canReplace: boolean
  feedback: LockFeedback | null
  onReplaceSlot: (team: Team, slotIndex: number) => void
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
            feedback?.team === team &&
            feedback.slotIndex === index

          return (
            <button
              className={`draft-slot ${isActive ? 'active' : ''} ${isLocked ? 'locked' : ''}`}
              disabled={!canReplace || !slot.championId || !slot.role}
              key={slot.player}
              onClick={() => onReplaceSlot(team, index)}
              type="button"
            >
              <span className={`role-sigil ${slot.role ? '' : 'empty'}`}>
                {slot.role ? positionIcon[slot.role] : ''}
              </span>
              <ChampionPortrait id={slot.championId} alt={champion?.name ?? `${slot.player} open pick`} />
              <span className="slot-copy">
                <strong>{slot.player}</strong>
                <span>{champion?.name ?? 'Open pick'}</span>
                {(slot.role || isActive) && <em>{slot.role ? roleLabel(slot.role) : 'Picking...'}</em>}
              </span>
              <span className="swap-icon" aria-hidden="true">
                {canReplace ? 'Edit' : 'Role'}
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
  const [stepIndex, setStepIndex] = useState(0)
  const [pendingChampionId, setPendingChampionId] = useState<string>()
  const [pendingRole, setPendingRole] = useState<Role>()
  const [replacementTarget, setReplacementTarget] = useState<ReplacementTarget>()
  const [feedback, setFeedback] = useState<LockFeedback | null>(null)
  const [search, setSearch] = useState('')
  const [roleFilter, setRoleFilter] = useState<Role | 'ALL'>('ALL')
  const [sortKey, setSortKey] = useState<SortKey>('winRate')
  const [modelOptionWinRates, setModelOptionWinRates] = useState<Record<string, number>>({})
  const [optionWinRatesLoading, setOptionWinRatesLoading] = useState(false)
  const [frozenChampionOptions, setFrozenChampionOptions] = useState<ChampionOption[] | null>(null)
  const lastRenderedChampionOptionsRef = useRef<ChampionOption[]>([])

  const currentStep = draftSteps[stepIndex]
  const isComplete = !currentStep
  const activeTeam = replacementTarget?.team ?? currentStep?.team ?? 'blue'
  const activeRole = replacementTarget?.role ?? pendingRole
  const activeSlotIndex = replacementTarget?.slotIndex ?? currentStep?.slotIndex
  const isReplacingPick = isComplete && !!replacementTarget
  const canSelectChampion = !isComplete || isReplacingPick
  const pendingChampion = championById(pendingChampionId)

  const [prediction, setPrediction] = useState<Prediction>(() =>
    localPrediction({
      bluePicks: initialBlueSlots,
      redPicks: initialRedSlots,
      activeTeam: 'blue',
      activeAction: 'pick',
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

      if (pendingChampionId && pendingRole && activeSlotIndex !== undefined && canSelectChampion) {
        const updateSlots = (slotsList: Slot[]) =>
          slotsList.map((slot, index) =>
            index === activeSlotIndex ? { ...slot, championId: pendingChampionId, role: pendingRole } : slot
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
        activeTeam,
        activeAction: 'pick',
        activeRole,
        stepIndex,
      }
    },
    [activeRole, activeTeam, blueSlots, canSelectChampion, redSlots, stepIndex, pendingChampionId, pendingRole, activeSlotIndex],
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
      const modelWinRate = modelOptionWinRates[optionId(champion.id, role)]
      if (modelWinRate !== undefined) {
        return {
          winRate: modelWinRate,
          sampleSize: local.sampleSize
        }
      }
      if (!champMetrics) return local

      const name = champion.name
      const realWr = champMetrics.winrates[name] !== undefined
        ? champMetrics.winrates[name] * 100
        : champMetrics.globalAvgWr * 100

      return {
        winRate: realWr,
        sampleSize: local.sampleSize
      }
    }
  }, [champMetrics, modelOptionWinRates])

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
              draft_state: toApiDraftPayload(draftPayload)
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

  const selectedChampionIds = useMemo(() => {
    const replacedChampionId = replacementTarget
      ? (replacementTarget.team === 'blue' ? blueSlots : redSlots)[replacementTarget.slotIndex]?.championId
      : undefined
    return new Set([
      ...blueSlots.map((slot) => slot.championId),
      ...redSlots.map((slot) => slot.championId),
    ].filter((championId): championId is string => !!championId && championId !== replacedChampionId))
  }, [blueSlots, redSlots, replacementTarget])

  const allowedOptionRoles = useMemo(() => {
    if (replacementTarget) return new Set<Role>([replacementTarget.role])
    const teamSlots = activeTeam === 'blue' ? blueSlots : redSlots
    const filledRoles = new Set(teamSlots.map((slot) => slot.role).filter((role): role is Role => !!role))
    return new Set(roles.filter((role) => !filledRoles.has(role)))
  }, [activeTeam, blueSlots, redSlots, replacementTarget])

  const filteredChampionOptions = useMemo<ChampionOption[]>(() => {
    const query = search.trim().toLowerCase()

    return champions
      .flatMap((champion) =>
        rolesForChampion(champion).map((role) => ({
          champion,
          entityId: optionId(champion.id, role),
          role,
        })),
      )
      .filter((option) => !selectedChampionIds.has(option.champion.id))
      .filter((option) => allowedOptionRoles.has(option.role))
      .filter((option) => roleFilter === 'ALL' || option.role === roleFilter)
      .filter((option) => {
        if (!query) return true
        return (
          option.champion.name.toLowerCase().includes(query) ||
          option.champion.title.toLowerCase().includes(query) ||
          option.champion.tags.join(' ').toLowerCase().includes(query) ||
          roleLabel(option.role).toLowerCase().includes(query)
        )
      })
  }, [allowedOptionRoles, roleFilter, search, selectedChampionIds])

  const visibleChampionOptions = useMemo<ChampionOption[]>(() => {
    return [...filteredChampionOptions].sort((a, b) => {
      const aMetrics = getRealMetrics(a.champion, a.role)
      const bMetrics = getRealMetrics(b.champion, b.role)
      if (sortKey === 'name') return a.champion.name.localeCompare(b.champion.name) || a.role.localeCompare(b.role)
      return bMetrics.winRate - aMetrics.winRate
    })
  }, [filteredChampionOptions, sortKey, getRealMetrics])

  const displayedChampionOptions = frozenChampionOptions ?? visibleChampionOptions

  const highlightedOptionIds = useMemo(() => {
    const bestByRole = new Map<Role, { entityId: string; winRate: number }>()

    for (const option of displayedChampionOptions) {
      if (!allowedOptionRoles.has(option.role)) continue

      const winRate = getRealMetrics(option.champion, option.role).winRate
      const currentBest = bestByRole.get(option.role)
      if (!currentBest || winRate > currentBest.winRate) {
        bestByRole.set(option.role, { entityId: option.entityId, winRate })
      }
    }

    return new Set([...bestByRole.values()].map((option) => option.entityId))
  }, [allowedOptionRoles, displayedChampionOptions, getRealMetrics])

  useEffect(() => {
    if (!canSelectChampion || activeSlotIndex === undefined) {
      setOptionWinRatesLoading(false)
      setFrozenChampionOptions(null)
      return
    }

    let cancelled = false

    if (filteredChampionOptions.length === 0) {
      setModelOptionWinRates({})
      setOptionWinRatesLoading(false)
      setFrozenChampionOptions(null)
      return
    }

    const draftState: DraftPayload = {
      bluePicks: blueSlots,
      redPicks: redSlots,
      activeTeam,
      activeAction: 'pick',
      stepIndex,
    }

    setFrozenChampionOptions(
      lastRenderedChampionOptionsRef.current.length > 0
        ? lastRenderedChampionOptionsRef.current
        : filteredChampionOptions,
    )
    setOptionWinRatesLoading(true)
    requestOptionWinRates(draftState, filteredChampionOptions)
      .then((result) => {
        if (!cancelled && result) {
          setModelOptionWinRates(result.optionWinRates)
        }
      })
      .catch((error) => {
        console.error('Batch option winrates failed:', error)
      })
      .finally(() => {
        if (!cancelled) {
          setFrozenChampionOptions(null)
          setOptionWinRatesLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [activeSlotIndex, activeTeam, blueSlots, canSelectChampion, filteredChampionOptions, redSlots, stepIndex])

  useEffect(() => {
    lastRenderedChampionOptionsRef.current = displayedChampionOptions
  }, [displayedChampionOptions])

  const statRows = useMemo(() => {
    return displayedChampionOptions.slice(0, 5).map((option) => ({
      champion: option.champion,
      metrics: getRealMetrics(option.champion, option.role),
      role: option.role,
    }))
  }, [displayedChampionOptions, getRealMetrics])

  function selectChampion(championId: string, role: Role) {
    if (!canSelectChampion || optionWinRatesLoading) return
    setPendingChampionId(championId)
    setPendingRole(role)
  }

  function confirmSelection() {
    if (!pendingChampionId || !pendingChampion || !pendingRole) return

    if (replacementTarget) {
      const updateSlots = (current: Slot[]) =>
        current.map((slot, index) =>
          index === replacementTarget.slotIndex
            ? { ...slot, championId: pendingChampionId, role: replacementTarget.role }
            : slot,
        )

      if (replacementTarget.team === 'blue') {
        setBlueSlots(updateSlots)
      } else {
        setRedSlots(updateSlots)
      }

      const stamp = window.Date.now()
      setFeedback({
        championName: pendingChampion.name,
        role: replacementTarget.role,
        slotIndex: replacementTarget.slotIndex,
        stamp,
        team: replacementTarget.team,
      })
      window.setTimeout(() => {
        setFeedback((current) => (current?.stamp === stamp ? null : current))
      }, 1100)
      setPendingChampionId(undefined)
      setPendingRole(undefined)
      setReplacementTarget(undefined)
      setRoleFilter('ALL')
      return
    }

    if (!currentStep) return

    if (currentStep.slotIndex !== undefined) {
      const updateSlots = (current: Slot[]) =>
        current.map((slot, index) =>
          index === currentStep.slotIndex ? { ...slot, championId: pendingChampionId, role: pendingRole } : slot,
        )

      if (currentStep.team === 'blue') {
        setBlueSlots(updateSlots)
      } else {
        setRedSlots(updateSlots)
      }
    }

    const stamp = window.Date.now()
    setFeedback({
      championName: pendingChampion.name,
      role: pendingRole,
      slotIndex: currentStep.slotIndex,
      stamp,
      team: currentStep.team,
    })
    window.setTimeout(() => {
      setFeedback((current) => (current?.stamp === stamp ? null : current))
    }, 1100)
    setPendingChampionId(undefined)
    setPendingRole(undefined)
    setStepIndex((current) => Math.min(current + 1, draftSteps.length))
  }

  function startReplacement(team: Team, slotIndex: number) {
    if (!isComplete) return
    const slot = (team === 'blue' ? blueSlots : redSlots)[slotIndex]
    if (!slot?.championId || !slot.role) return
    setReplacementTarget({ team, slotIndex, role: slot.role })
    setPendingChampionId(undefined)
    setPendingRole(slot.role)
    setSearch('')
    setRoleFilter(slot.role)
  }

  function cancelReplacement() {
    setReplacementTarget(undefined)
    setPendingChampionId(undefined)
    setPendingRole(undefined)
    setRoleFilter('ALL')
  }

  function resetDraft() {
    setBlueSlots(initialBlueSlots)
    setRedSlots(initialRedSlots)
    setStepIndex(0)
    setPendingChampionId(undefined)
    setPendingRole(undefined)
    setReplacementTarget(undefined)
    setFeedback(null)
    setModelOptionWinRates({})
    setOptionWinRatesLoading(false)
    setPrediction(localPrediction({
      bluePicks: initialBlueSlots,
      redPicks: initialRedSlots,
      activeTeam: 'blue',
      activeAction: 'pick',
      stepIndex: 0,
    }))
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
                  ? `${teamName(currentStep.team)} is picking`
                  : 'All picks are locked'}
              </em>
            </div>
            <div className="header-tools" aria-label="Settings">
              <button type="button" onClick={resetDraft} aria-label="Reset draft">Reset</button>
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
              canReplace={isComplete}
              feedback={feedback}
              onReplaceSlot={startReplacement}
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
                    <option value="name">Name</option>
                  </select>
                  {optionWinRatesLoading && <em>Updating</em>}
                </label>
              </div>

              <div className={`action-banner ${activeTeam}`}>
                <div className="action-copy">
                  <strong>
                    {replacementTarget
                      ? `${teamName(replacementTarget.team)} ${roleLabel(replacementTarget.role)} replacement`
                      : currentStep?.label ?? 'Draft complete'}
                  </strong>
                  <span>
                    {replacementTarget
                      ? `Choose a new ${roleLabel(replacementTarget.role)} champion for this player.`
                      : isComplete
                        ? 'Click a player box to replace that champion.'
                        : 'Select a champion, then confirm the pick.'}
                  </span>
                </div>
                <div className="pending-selection">
                  <ChampionPortrait id={pendingChampionId} alt={pendingChampion?.name ?? 'No selected champion'} />
                  <span>
                    <strong>{pendingChampion?.name ?? 'No champion selected'}</strong>
                    <em>
                      {replacementTarget
                        ? `${teamName(replacementTarget.team)} ${roleLabel(replacementTarget.role)}`
                        : currentStep
                          ? `${teamName(currentStep.team)} ${pendingRole ? roleLabel(pendingRole) : 'pick'}`
                          : 'Draft complete'}
                    </em>
                  </span>
                  {replacementTarget ? (
                    <div className="pending-buttons">
                      <button disabled={!pendingChampionId} onClick={confirmSelection} type="button">Replace</button>
                      <button className="secondary" onClick={cancelReplacement} type="button">Cancel</button>
                    </div>
                  ) : isComplete ? (
                    <button onClick={resetDraft} type="button">Reset draft</button>
                  ) : (
                    <button disabled={!pendingChampionId || !pendingRole} onClick={confirmSelection} type="button">
                      Confirm pick
                    </button>
                  )}
                </div>
              </div>

              {isComplete && !replacementTarget && (
                <div className="final-actions" aria-label="Finished draft actions">
                  <div>
                    <strong>Final draft ready</strong>
                    <span>Click any filled player box to choose another champion for that same role.</span>
                  </div>
                </div>
              )}

              {feedback && (
                <div className={`lock-feedback ${feedback.team}`} key={feedback.stamp}>
                  <strong>{feedback.championName}</strong>
                  <span>{teamName(feedback.team)} {roleLabel(feedback.role)} locked</span>
                </div>
              )}

              <div className={`champion-grid ${optionWinRatesLoading ? 'is-loading' : ''}`} aria-busy={optionWinRatesLoading}>
                {displayedChampionOptions.length === 0 && (
                  <div className="champion-card placeholder">No matches</div>
                )}
                {displayedChampionOptions.map(({ champion, entityId, role }) => {
                  const metrics = getRealMetrics(champion, role)
                  const isHighlighted = highlightedOptionIds.has(entityId)
                  const isSelected = pendingChampionId === champion.id && pendingRole === role

                  return (
                    <button
                      className={`champion-card ${isSelected ? 'selected' : ''} ${isHighlighted ? 'recommended' : ''}`}
                      disabled={!canSelectChampion || optionWinRatesLoading}
                      key={entityId}
                      onClick={() => selectChampion(champion.id, role)}
                      type="button"
                    >
                      {isHighlighted && <span className="favorite">*</span>}
                      <ChampionPortrait id={champion.id} alt={champion.name} />
                      <span className="champion-card-copy">
                        <strong>
                          {champion.name}
                          <em className="card-role-tag">{roleLabel(role)}</em>
                        </strong>
                        <span>{metrics.winRate.toFixed(1)}%</span>
                      </span>
                      <i aria-hidden="true">Pick</i>
                    </button>
                  )
                })}
                {optionWinRatesLoading && (
                  <div className="pool-loading" role="status" aria-live="polite">
                    <span className="pool-loading-spinner" aria-hidden="true" />
                    <strong>Updating winrates</strong>
                  </div>
                )}
              </div>
            </section>

            <TeamPanel
              activeSlotIndex={activeSlotIndex}
              activeTeam={activeTeam}
              canReplace={isComplete}
              feedback={feedback}
              onReplaceSlot={startReplacement}
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
                  <span>Sample Size</span>
                </div>
                {statRows.map(({ champion, metrics, role }) => (
                  <div className="stats-row" key={optionId(champion.id, role)} role="row">
                    <span className="stat-champion">
                      <ChampionPortrait id={champion.id} alt={champion.name} />
                      {champion.name}
                    </span>
                    <span>{roleLabel(role)}</span>
                    <strong>{metrics.winRate.toFixed(1)}%</strong>
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
                      <p>Pick a champion to trigger tactical co-pilot advice.</p>
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

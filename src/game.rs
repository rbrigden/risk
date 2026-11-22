use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

pub const REGIONS: [[usize; 7]; 6] = [
    [0, 1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12, 13],
    [14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27],
    [28, 29, 30, 31, 32, 33, 34],
    [35, 36, 37, 38, 39, 40, 41],
];
const REGION_SIZE: usize = REGIONS[0].len();
pub const N: usize = REGIONS.len() * REGION_SIZE;
#[cfg(not(target_arch = "wasm32"))]
pub const MAX_REINFORCEMENTS: u64 = reinforcements(N, REGIONS.len());

const fn reinforcements(owned: usize, regions: usize) -> u64 {
    (2 + owned / 4 + 2 * regions) as u64
}
pub const MAX_PLAYERS: usize = 6;
pub const BRIDGES: [(usize, usize); 9] = [
    (1, 11),
    (8, 18),
    (22, 32),
    (29, 39),
    (3, 21),
    (10, 28),
    (17, 35),
    (2, 33),
    (9, 40),
];
const EDGE_COUNT: usize = 2 * (12 * REGIONS.len() + BRIDGES.len());

const fn edges() -> [(usize, usize); EDGE_COUNT] {
    let mut edges = [(0, 0); EDGE_COUNT];
    let mut count = 0;
    let mut a = 0;
    while a < N {
        let mut b = 0;
        while b < N {
            let coast = REGION_SIZE - 1;
            let (x, y) = (a % REGION_SIZE, b % REGION_SIZE);
            let mut connected = a / REGION_SIZE == b / REGION_SIZE
                && x != y
                && (x == coast || y == coast || x.abs_diff(y) == 1 || x.abs_diff(y) == coast - 1);
            let mut i = 0;
            while i < BRIDGES.len() {
                let (x, y) = BRIDGES[i];
                connected |= (a == x && b == y) || (a == y && b == x);
                i += 1;
            }
            if connected {
                edges[count] = (a, b);
                count += 1;
            }
            b += 1;
        }
        a += 1;
    }
    assert!(count == edges.len());
    edges
}

pub const EDGES: [(usize, usize); EDGE_COUNT] = edges();
const FORTIFY_START: usize = N + 3 * EDGES.len();
const FINISH_ATTACK: usize = FORTIFY_START + 3 * EDGES.len();
pub const ACTION_COUNT: usize = FINISH_ATTACK + 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Place(usize),
    Attack(usize, usize, usize),
    Fortify(usize, usize, usize),
    FinishAttack,
    PassFortify,
}

impl Action {
    pub fn decode(id: usize) -> Option<Self> {
        let edge_action = |offset: usize| {
            let (a, b) = EDGES[offset / 3];
            (a, b, offset % 3)
        };
        Some(match id {
            0..N => Self::Place(id),
            N..FORTIFY_START => {
                let (a, b, option) = edge_action(id - N);
                Self::Attack(a, b, option + 1)
            }
            FORTIFY_START..FINISH_ATTACK => {
                let (a, b, option) = edge_action(id - FORTIFY_START);
                Self::Fortify(a, b, option)
            }
            FINISH_ATTACK => Self::FinishAttack,
            id if id == FINISH_ATTACK + 1 => Self::PassFortify,
            _ => return None,
        })
    }

    pub fn encode(self) -> Option<usize> {
        let edge_id = |a, b| EDGES.binary_search(&(a, b)).ok().map(|i| 3 * i);
        match self {
            Self::Place(a) if a < N => Some(a),
            Self::Attack(a, b, dice @ 1..=3) => Some(N + edge_id(a, b)? + dice - 1),
            Self::Fortify(a, b, option @ 0..=2) => Some(FORTIFY_START + edge_id(a, b)? + option),
            Self::FinishAttack => Some(FINISH_ATTACK),
            Self::PassFortify => Some(FINISH_ATTACK + 1),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Reinforce = 0,
    Attack = 1,
    Fortify = 2,
    Ended = 3,
}

#[cfg_attr(not(feature = "visualizer"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct BattleOutcome {
    pub attack: [u8; 3],
    pub defense: [u8; 2],
    pub attacker_losses: u64,
    pub defender_losses: u64,
    pub captured: bool,
}

pub struct Game {
    pub players: usize,
    pub max_turns: u32,
    pub owner: [usize; N],
    pub troops: [u64; N],
    pub active: usize,
    pub phase: Phase,
    pub pending: u64,
    pub turns: u32,
    pub winner: Option<usize>,
    rng: ChaCha8Rng,
}

impl Game {
    pub fn new(players: usize, max_turns: u32, seed: Option<u64>) -> Self {
        let mut rng = seed.map_or_else(ChaCha8Rng::from_os_rng, ChaCha8Rng::seed_from_u64);
        let active = rng.random_range(0..players);
        let mut territories: [usize; N] = std::array::from_fn(|i| i);
        territories.shuffle(&mut rng);
        let mut owner = [0; N];
        for (i, territory) in territories.into_iter().enumerate() {
            owner[territory] = (active + i) % players;
        }
        let mut game = Self {
            players,
            max_turns,
            owner,
            troops: [1; N],
            active,
            phase: Phase::Reinforce,
            pending: 0,
            turns: 0,
            winner: None,
            rng,
        };
        game.start_turn();
        game
    }

    fn start_turn(&mut self) {
        let owned = self.owner.iter().filter(|&&p| p == self.active).count();
        let regions = REGIONS
            .iter()
            .filter(|region| region.iter().all(|&i| self.owner[i] == self.active))
            .count();
        self.pending = reinforcements(owned, regions);
        self.phase = Phase::Reinforce;
    }

    fn end_turn(&mut self) {
        self.turns += 1;
        if self.turns == self.max_turns {
            self.phase = Phase::Ended;
            return;
        }
        loop {
            self.active = (self.active + 1) % self.players;
            if self.owner.contains(&self.active) {
                break;
            }
        }
        self.start_turn();
    }

    pub fn fortify_amount(&self, source: usize, option: usize) -> Option<u64> {
        let movable = self.troops[source] - 1;
        match option {
            0 if movable >= 1 => Some(1),
            1 if movable / 2 > 1 => Some(movable / 2),
            2 if movable > 1 => Some(movable),
            _ => None,
        }
    }

    pub fn legal(&self, action: Action) -> bool {
        let owned = |a: usize| self.owner[a] == self.active;
        match (self.phase, action) {
            (Phase::Reinforce, Action::Place(a)) => owned(a),
            (Phase::Attack, Action::Attack(a, b, dice)) => {
                owned(a) && !owned(b) && self.troops[a] > dice as u64
            }
            (Phase::Fortify, Action::Fortify(a, b, option)) => {
                owned(a) && owned(b) && self.fortify_amount(a, option).is_some()
            }
            (Phase::Attack, Action::FinishAttack) | (Phase::Fortify, Action::PassFortify) => true,
            _ => false,
        }
    }

    pub fn step(&mut self, action: Action) -> Option<BattleOutcome> {
        debug_assert!(self.legal(action));
        match action {
            Action::Place(a) => {
                self.troops[a] += 1;
                self.pending -= 1;
                if self.pending == 0 {
                    self.phase = Phase::Attack;
                }
            }
            Action::Attack(a, b, dice) => {
                let mut attack = [0; 3];
                let mut defense = [0; 2];
                let defense_dice = self.troops[b].min(2) as usize;
                for die in attack[..dice]
                    .iter_mut()
                    .chain(&mut defense[..defense_dice])
                {
                    *die = self.rng.random_range(1..=6);
                }
                return Some(self.resolve_attack(
                    a,
                    b,
                    &mut attack[..dice],
                    &mut defense[..defense_dice],
                ));
            }
            Action::Fortify(a, b, option) => {
                let moved = self.fortify_amount(a, option).unwrap();
                self.troops[a] -= moved;
                self.troops[b] += moved;
                self.end_turn();
            }
            Action::FinishAttack => self.phase = Phase::Fortify,
            Action::PassFortify => self.end_turn(),
        }
        None
    }

    fn resolve_attack(
        &mut self,
        a: usize,
        b: usize,
        attack: &mut [u8],
        defense: &mut [u8],
    ) -> BattleOutcome {
        attack.sort_unstable_by(|a, b| b.cmp(a));
        defense.sort_unstable_by(|a, b| b.cmp(a));
        let mut result = BattleOutcome {
            attack: [0; 3],
            defense: [0; 2],
            attacker_losses: 0,
            defender_losses: 0,
            captured: false,
        };
        result.attack[..attack.len()].copy_from_slice(attack);
        result.defense[..defense.len()].copy_from_slice(defense);
        for (&attacker, &defender) in attack.iter().zip(defense.iter()) {
            if attacker > defender {
                result.defender_losses += 1;
            } else {
                result.attacker_losses += 1;
            }
        }
        self.troops[a] -= result.attacker_losses;
        self.troops[b] -= result.defender_losses;
        result.captured = self.troops[b] == 0;
        if result.captured {
            let moved = attack.len() as u64;
            self.owner[b] = self.active;
            self.troops[a] -= moved;
            self.troops[b] = moved;
            if self.owner.iter().all(|&p| p == self.active) {
                self.winner = Some(self.active);
                self.phase = Phase::Ended;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connected_board_and_action_round_trips() {
        let mut visited = [false; N];
        visited[0] = true;
        for _ in 0..N {
            for (a, b) in EDGES {
                visited[b] |= visited[a];
            }
        }
        assert!(visited.into_iter().all(|v| v));
        assert_eq!(ACTION_COUNT, 1016);
        for (a, b) in EDGES {
            assert!(EDGES.binary_search(&(b, a)).is_ok());
        }
        for id in 0..ACTION_COUNT {
            assert_eq!(Action::decode(id).unwrap().encode(), Some(id));
        }
        assert_eq!(Action::decode(ACTION_COUNT), None);
    }

    #[test]
    fn setup_and_turn_order() {
        for players in 2..=MAX_PLAYERS {
            let mut game = Game::new(players, 10, Some(19));
            let starting_seat = game.active;
            for i in 0..players {
                let seat = (starting_seat + i) % players;
                let owned = game.owner.iter().filter(|&&p| p == seat).count();
                assert_eq!(owned, N / players + usize::from(i < N % players));
            }
            assert_eq!(game.troops, [1; N]);
            game.end_turn();
            assert_eq!(game.active, (starting_seat + 1) % players);
        }
    }

    #[test]
    fn reinforcement_and_fortification() {
        let mut game = Game::new(2, 2, Some(0));
        game.active = 0;
        game.owner = std::array::from_fn(|i| usize::from(i >= REGION_SIZE));
        game.start_turn();
        assert_eq!(game.pending, 5);
        for _ in 0..5 {
            game.step(Action::Place(0));
        }
        game.step(Action::FinishAttack);
        assert_eq!(game.fortify_amount(0, 0), Some(1));
        assert_eq!(game.fortify_amount(0, 1), Some(2));
        assert_eq!(game.fortify_amount(0, 2), Some(5));
        assert_eq!(game.fortify_amount(1, 0), None);
        game.step(Action::Fortify(0, 1, 0));
        assert_eq!((game.troops[0], game.troops[1]), (5, 2));
        game.end_turn();
        assert_eq!(game.phase, Phase::Ended);
        assert_eq!(game.winner, None);
    }

    #[test]
    fn reports_split_losses_in_comparison_order() {
        let mut game = Game::new(2, 10, Some(0));
        game.troops[0] = 4;
        game.troops[1] = 2;
        let result = game.resolve_attack(0, 1, &mut [1, 6, 2], &mut [3, 5]);
        assert_eq!(result.attack, [6, 2, 1]);
        assert_eq!(result.defense, [5, 3]);
        assert_eq!((result.attacker_losses, result.defender_losses), (1, 1));
        assert!(!result.captured);
        assert_eq!((game.troops[0], game.troops[1]), (3, 1));
    }

    #[test]
    fn ties_capture_elimination_and_victory() {
        let mut game = Game::new(3, 10, Some(0));
        game.active = 0;
        game.owner = [0; N];
        game.owner[1] = 1;
        game.owner[2] = 2;
        game.troops[0] = 5;
        let tie = game.resolve_attack(0, 1, &mut [5, 4, 2], &mut [5]);
        assert_eq!((tie.attacker_losses, tie.defender_losses), (1, 0));
        assert_eq!((game.troops[0], game.troops[1]), (4, 1));
        let capture = game.resolve_attack(0, 1, &mut [2, 6, 3], &mut [5]);
        assert_eq!(capture.attack, [6, 3, 2]);
        assert_eq!(capture.defense, [5, 0]);
        assert_eq!((capture.attacker_losses, capture.defender_losses), (0, 1));
        assert!(capture.captured);
        assert_eq!((game.troops[0], game.troops[1]), (1, 3));
        assert_eq!(game.owner[1], 0);
        game.end_turn();
        assert_eq!(game.active, 2);
        game.end_turn();
        game.resolve_attack(1, 2, &mut [6, 5], &mut [1]);
        assert_eq!(game.winner, Some(0));
        assert_eq!(game.phase, Phase::Ended);
    }
}

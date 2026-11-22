use crate::game::{
    ACTION_COUNT, Action, EDGES, Game, MAX_PLAYERS, MAX_REINFORCEMENTS, N, Phase, REGIONS,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

type ActionFields<'a> = (&'a str, Option<usize>, Option<usize>, Option<usize>);

fn invalid_action() -> PyErr {
    PyValueError::new_err("invalid action")
}

#[pyfunction]
#[pyo3(signature = (kind, source=None, target=None, option=None))]
fn encode_action(
    kind: &str,
    source: Option<usize>,
    target: Option<usize>,
    option: Option<usize>,
) -> PyResult<usize> {
    let action = match (kind, source, target, option) {
        ("place", Some(a), None, None) => Action::Place(a),
        ("attack", Some(a), Some(b), Some(dice)) => Action::Attack(a, b, dice),
        ("fortify", Some(a), Some(b), Some(amount)) => Action::Fortify(a, b, amount),
        ("finish_attack", None, None, None) => Action::FinishAttack,
        ("pass_fortify", None, None, None) => Action::PassFortify,
        _ => return Err(invalid_action()),
    };
    action.encode().ok_or_else(invalid_action)
}

#[pyfunction]
fn decode_action(action_id: &Bound<'_, PyAny>) -> PyResult<ActionFields<'static>> {
    Ok(match parse_action(action_id)? {
        Action::Place(a) => ("place", Some(a), None, None),
        Action::Attack(a, b, dice) => ("attack", Some(a), Some(b), Some(dice)),
        Action::Fortify(a, b, option) => ("fortify", Some(a), Some(b), Some(option)),
        Action::FinishAttack => ("finish_attack", None, None, None),
        Action::PassFortify => ("pass_fortify", None, None, None),
    })
}

fn parse_action(action_id: &Bound<'_, PyAny>) -> PyResult<Action> {
    action_id
        .extract::<usize>()
        .ok()
        .and_then(Action::decode)
        .ok_or_else(invalid_action)
}

#[pyclass(module = "riskbot")]
struct Simulator {
    game: Game,
}

#[pymethods]
impl Simulator {
    #[new]
    #[pyo3(signature = (num_players=2, seed=None, max_turns=500))]
    fn new(num_players: i64, seed: Option<u64>, max_turns: i64) -> PyResult<Self> {
        if !(2..=MAX_PLAYERS as i64).contains(&num_players) {
            return Err(PyValueError::new_err("num_players must be between 2 and 6"));
        }
        let max_turns = u32::try_from(max_turns)
            .ok()
            .filter(|&turns| turns > 0)
            .ok_or_else(|| PyValueError::new_err("max_turns must be between 1 and 4294967295"))?;
        Ok(Self {
            game: Game::new(num_players as usize, max_turns, seed),
        })
    }

    #[pyo3(signature = (seed=None))]
    fn reset<'py>(&mut self, py: Python<'py>, seed: Option<u64>) -> PyResult<Bound<'py, PyDict>> {
        self.game = Game::new(self.game.players, self.game.max_turns, seed);
        self.observation(py, self.game.active)
    }

    fn observe<'py>(&self, py: Python<'py>, player: i64) -> PyResult<Bound<'py, PyDict>> {
        if !(0..self.game.players as i64).contains(&player) {
            return Err(PyValueError::new_err(
                "player is outside the configured seats",
            ));
        }
        self.observation(py, player as usize)
    }

    #[getter]
    fn active_player(&self) -> Option<usize> {
        (self.game.phase != Phase::Ended).then_some(self.game.active)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action_id: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyTuple>)> {
        if self.game.phase == Phase::Ended {
            return Err(PyRuntimeError::new_err("episode has ended"));
        }
        let action = parse_action(action_id)?;
        if !self.game.legal(action) {
            return Err(invalid_action());
        }
        let actor = self.game.active;
        self.game.step(action);
        let observer = self.active_player().unwrap_or(actor);
        let rewards = PyTuple::new(
            py,
            (0..self.game.players).map(|player| match self.game.winner {
                Some(winner) if player == winner => 1,
                Some(_) => -1,
                None => 0,
            }),
        )?;
        Ok((self.observation(py, observer)?, rewards))
    }

    fn observation_spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let spec = PyDict::new(py);
        let last_seat = self.game.players as i64 - 1;
        let troop_limit = N as i64 + MAX_REINFORCEMENTS as i64 * i64::from(self.game.max_turns);
        for (key, length, minimum, maximum) in [
            ("owners", Some(N), 0, last_seat),
            ("troops", Some(N), 1, troop_limit),
            ("alive", Some(MAX_PLAYERS), 0, 1),
            ("player", None, 0, last_seat),
            ("active_player", None, -1, last_seat),
            ("phase", None, 0, 3),
            ("reinforcements", None, 0, MAX_REINFORCEMENTS as i64),
            ("completed_turns", None, 0, i64::from(self.game.max_turns)),
            ("winner", None, -1, last_seat),
            ("terminated", None, 0, 1),
            ("truncated", None, 0, 1),
            ("action_mask", Some(ACTION_COUNT), 0, 1),
        ] {
            spec.set_item(key, (length, minimum, maximum))?;
        }
        Ok(spec)
    }
}

impl Simulator {
    fn observation<'py>(&self, py: Python<'py>, player: usize) -> PyResult<Bound<'py, PyDict>> {
        let game = &self.game;
        let active = self.active_player();
        let observation = PyDict::new(py);
        observation.set_item("owners", PyTuple::new(py, game.owner)?)?;
        observation.set_item("troops", PyTuple::new(py, game.troops)?)?;
        observation.set_item(
            "alive",
            PyTuple::new(py, (0..MAX_PLAYERS).map(|p| game.owner.contains(&p)))?,
        )?;
        observation.set_item("player", player)?;
        observation.set_item("active_player", active.map_or(-1, |p| p as i64))?;
        observation.set_item("phase", game.phase as u8)?;
        observation.set_item("reinforcements", game.pending)?;
        observation.set_item("completed_turns", game.turns)?;
        observation.set_item("winner", game.winner.map_or(-1, |p| p as i64))?;
        observation.set_item("terminated", game.winner.is_some())?;
        observation.set_item("truncated", active.is_none() && game.winner.is_none())?;
        let mask = (0..ACTION_COUNT)
            .map(|id| active == Some(player) && game.legal(Action::decode(id).unwrap()));
        observation.set_item("action_mask", PyTuple::new(py, mask)?)?;
        Ok(observation)
    }
}

#[pymodule]
fn riskbot(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Simulator>()?;
    m.add_function(wrap_pyfunction!(encode_action, m)?)?;
    m.add_function(wrap_pyfunction!(decode_action, m)?)?;
    m.add("ACTION_COUNT", ACTION_COUNT)?;
    m.add("DIRECTED_EDGES", PyTuple::new(m.py(), EDGES)?)?;
    m.add(
        "REGIONS",
        PyTuple::new(
            m.py(),
            REGIONS
                .into_iter()
                .map(|region| PyTuple::new(m.py(), region))
                .collect::<PyResult<Vec<_>>>()?,
        )?,
    )?;
    Ok(())
}

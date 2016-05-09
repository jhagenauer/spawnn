package heuristics.tabu;

import java.util.List;

import heuristics.Individual;

public abstract class TabuIndividual<T> extends Individual<T> {
	public abstract TabuIndividual<T> applyMove( TabuMove<T> tm );
	public abstract TabuMove<T> getRandomMove();
	public abstract List<TabuMove<T>> getNeighboringMoves();
}

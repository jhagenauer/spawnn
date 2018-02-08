package heuristics.tabu;

import java.util.List;

import heuristics.HeuristicsIndividual;

public abstract class TabuIndividual<T> implements HeuristicsIndividual {
	public abstract TabuIndividual<T> applyMove( TabuMove<T> tm );
	public abstract TabuMove<T> getRandomMove();
	public abstract List<TabuMove<T>> getNeighboringMoves();
	
	private double value = 0;
	
	public void setValue(double value) {
		this.value = value;
	}
	
	public double getValue() {
		return value;
	}
	
	public int compareTo(T o) {
		return Double.compare(value, ((TabuIndividual<T>) o).getValue());
	}
}

package heuristics.sa;

import heuristics.HeuristicsIndividual;

public interface SAIndividual<T> extends HeuristicsIndividual {
	public void step();
	public T getCopy();
}

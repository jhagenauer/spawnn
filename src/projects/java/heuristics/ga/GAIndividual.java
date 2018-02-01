package heuristics.ga;

import heuristics.Individual;

public interface GAIndividual<T> extends Individual<T> {
	public T mutate();
	public T recombine( T mother );
}

package ga;

import heuristics.HeuristicsIndividual;

public interface GAIndividual<T> extends HeuristicsIndividual {
	public T mutate();
	public T recombine( T mother );
}

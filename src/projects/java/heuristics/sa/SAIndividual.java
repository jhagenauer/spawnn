package heuristics.sa;

import heuristics.Individual;

public interface SAIndividual<T> extends Individual<T> {
	public void step();
	public T getCopy();
}

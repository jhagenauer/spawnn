package heuristics.tabu;

public interface Evaluator<T extends TabuIndividual<T>> {
	public double evaluate(T i);
}

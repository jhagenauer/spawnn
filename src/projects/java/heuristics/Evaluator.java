package heuristics;

public interface Evaluator<T extends Individual<T>> {
	public double evaluate(T i);
}

package myga;

public interface Evaluator<T extends GAIndividual<T>> {
	public double evaluate(T i);
}

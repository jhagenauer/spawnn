package myga;

public interface Evaluator<T extends GAIndividual> {
	public double evaluate(T i);
}

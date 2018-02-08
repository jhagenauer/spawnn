package heuristics;

public interface CostCalculator<T extends HeuristicsIndividual> {
	public double getCost(T i);
}

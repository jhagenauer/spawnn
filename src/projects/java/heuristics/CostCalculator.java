package heuristics;

public interface CostCalculator<T extends Individual<?>> {
	public double getCost(T i);
}

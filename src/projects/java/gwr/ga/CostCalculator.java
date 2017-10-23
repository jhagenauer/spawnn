package gwr.ga;

public interface CostCalculator<T extends GAIndividual> {
	public double getCost(T i);
}

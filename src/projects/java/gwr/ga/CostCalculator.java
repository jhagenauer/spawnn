package gwr.ga;

public interface CostCalculator<T extends GAIndividual<T>> {
	public double getCost(T i);
}

package regionalization.ga;

public abstract interface GAIndividual extends Comparable<GAIndividual>{
	public abstract GAIndividual mutate();
	public abstract GAIndividual recombine( GAIndividual mother );
	public double getValue();
}

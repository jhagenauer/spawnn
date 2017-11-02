package gwr.ga;

public interface GAIndividual {
	public GAIndividual mutate();
	public GAIndividual recombine( GAIndividual mother );
}

package moomap.myga;

//TODO make class abstract, implement value-stuff 
public abstract interface GAIndividual extends Comparable<GAIndividual>{
	public abstract void mutate();
	public abstract GAIndividual recombine( GAIndividual mother );
	public void setValue(double value);
	public double getValue();
}

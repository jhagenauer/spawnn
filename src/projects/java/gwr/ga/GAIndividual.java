package gwr.ga;

public abstract class GAIndividual implements Comparable<GAIndividual>{
		
	public abstract GAIndividual mutate();
	public abstract GAIndividual recombine( GAIndividual mother );
	public abstract double getCost();	
	
	@Override
	public int compareTo(GAIndividual o) {
		if (getCost() < o.getCost())
			return -1;
		else if (getCost() > o.getCost())
			return 1;
		else
			return 0;
	}
}

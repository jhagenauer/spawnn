package heuristics;

public abstract class GAIndividual<T> extends Individual<T>{
	public abstract void mutate();
	public abstract T recombine( T mother );
}

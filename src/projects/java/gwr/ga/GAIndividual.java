package gwr.ga;

public interface GAIndividual<T> {
	public T mutate();
	public T recombine( T mother );
}

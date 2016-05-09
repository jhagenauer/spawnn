package heuristics.tabu;

import java.util.HashMap;
import java.util.Map;

import org.apache.log4j.Logger;

public class TabuList<E> {
	
	protected static Logger log = Logger.getLogger(TabuList.class);
	
	private int iter;
	private Map<E,int[]> map;
	
	public TabuList() {
		iter = 0;
		map = new HashMap<E,int[]>();
	}
	
	public void add( E e, int tenure ) {
		map.put( e, new int[]{ iter, tenure } );
	}
	
	public int getTenure( E e ) {
		return map.get(e)[1];
	}
	
	public boolean isTabu( E e ) {
		if( map.containsKey( e ) ) {
			int[] i = (int[])map.get(e);
			if( i[1] + i[0]  < iter ) 
				return false;
			else
				return true;
		} else
			return false;
	}
	
	public void step() {
		iter++;
	}
	
	public int size() {
		return map.size();
	}
	
	public String toString() {
		return map.toString();
	}
}

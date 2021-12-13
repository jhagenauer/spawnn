package spawnn.som.bmu;


import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import spawnn.ng.sorter.SorterContext;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public class SorterBmuGetterContext extends BmuGetterContext {
	
	private static Logger log = LogManager.getLogger(SorterBmuGetterContext.class);
	
	private SorterContext s;
	
	public SorterBmuGetterContext( SorterContext s ) {
		this.s = s;
	}

	@Override
	public GridPos getBmuPos( double[] x, Grid<double[]> grid, Set<GridPos> ign ) { // slow
		
		if( !ign.isEmpty() )
			log.warn("Ign not supported yet!"); // because it requires new update-hist-strategy
		
		List<double[]> l = new ArrayList<double[]>(grid.getPrototypes());
		s.sort(x, l);
		return grid.getPositionOf(l.get(0));
	}

	@Override
	public double[] getContext(double[] x) {
		return s.getContext(x);
	}
}

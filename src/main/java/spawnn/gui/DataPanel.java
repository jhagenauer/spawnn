package spawnn.gui;

import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.swing.DefaultCellEditor;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.DefaultTableModel;

import net.miginfocom.swing.MigLayout;

import org.apache.log4j.Logger;

import spawnn.dist.EuclideanDist;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

public class DataPanel extends JPanel implements ActionListener, TableModelListener {
	
	private static Logger log = Logger.getLogger(DataPanel.class);
	private static final long serialVersionUID = 1736311423918203189L;
	/* Training allowed, if one variable at least AND shapefile OR coordinates are marked
	 * centroid-models allowed only if coordinates are marked */
	public static final String TRAIN_ALLOWED_PROP ="TRAIN_ALLOWED";
	static Set<String> coordNames;
	private boolean useAllState = false;
		
	static {
		coordNames = new HashSet<String>();
		coordNames.add("x");
		coordNames.add("X");
		coordNames.add("y");
		coordNames.add("Y");
		coordNames.add("lon");
		coordNames.add("LON");
		coordNames.add("Lon");
		coordNames.add("longitude");
		coordNames.add("lat");
		coordNames.add("LAT");
		coordNames.add("Lat");
		coordNames.add("latitude");
	}
		
	private DefaultTableModel dataTable;
	private JButton btnLoadShp, addCCoords, useAll;
	private BoxPlotPanel boxPlotPnl;
	private Frame parent;
	
	enum norm {none, scale, zScore};
	final int ATTRIBUTE = 0, NORM = 1, COORDINATE = 2, USE = 3;
	
	protected SpatialDataFrame sd = null;
	
	DataPanel(Frame parent) {
		super();
		this.parent = parent;
		
		setLayout(new MigLayout(""));
		
		btnLoadShp = new JButton("Load data...");
		btnLoadShp.addActionListener(this);
		
		addCCoords = new JButton("Add centroid coordinates");
		addCCoords.addActionListener(this);
		addCCoords.setEnabled(false);
				
		useAll = new JButton("Use all");
		useAll.addActionListener(this);
		useAll.setEnabled(false);
						
		dataTable = new DefaultTableModel( new String[]{"Attribute","Normalize","Coordinate","Use"}, 0 ) {
			private static final long serialVersionUID = -78686917074445663L;

			@Override
			public Class<?> getColumnClass(int column) {
				switch (column) {
                case ATTRIBUTE:
                    return String.class;
                case NORM:
                    return String.class;
                case COORDINATE:
                    return Boolean.class;
                case USE:
                    return Boolean.class;
                default:
                	return String.class;
				}
			}
		};
		
		dataTable.addTableModelListener(this);
		
		final JTable table = new JTable(dataTable);
		table.getColumnModel().getColumn(1).setCellEditor(new DefaultCellEditor( new JComboBox(norm.values())));
		
		table.getColumnModel().getColumn(0).setPreferredWidth(250);
		table.getColumnModel().getColumn(1).setPreferredWidth(20);
		table.getColumnModel().getColumn(2).setPreferredWidth(10);
		table.getColumnModel().getColumn(3).setPreferredWidth(10);
				
		boxPlotPnl = new BoxPlotPanel();
		
		add(btnLoadShp, "split 4");
		add(addCCoords, "");
		add(useAll,"wrap");
		
		add( new JScrollPane(table), "w 50%, grow" );
		add( boxPlotPnl, "w 50%, push, grow");
	}

	@Override
	public void actionPerformed(ActionEvent ae) {
		
		if( ae.getSource() == btnLoadShp ) {
			JFileChooser fc = new JFileChooser();
					
			fc.setFileFilter(FFilter.csvFilter); 
			fc.setFileFilter(FFilter.shpFilter);
									
			int state = fc.showOpenDialog(this);
			if( state == JFileChooser.APPROVE_OPTION ) {
				File fn = fc.getSelectedFile(); 
				
				if( fc.getFileFilter() == FFilter.shpFilter ) {
			      sd = DataUtils.readSpatialDataFrameFromShapefile(fn, false);     
			      addCCoords.setEnabled(true);
				} else if( fc.getFileFilter() == FFilter.csvFilter ) {				
					DataFrame df = DataUtils.readDataFrameFromCSV(fn, new int[]{}, true ); 
					sd = new SpatialDataFrame();				
					sd.samples = df.samples;
					sd.names = df.names;
					sd.bindings = df.bindings;
				}
				useAll.setEnabled(true);
				
				dataTable.setRowCount(0);
								
				for( String s : sd.names )
			    	if( coordNames.contains(s) )
			    		dataTable.addRow( new Object[]{s, norm.none, true, false } );
			    	else
			    		dataTable.addRow( new Object[]{s, norm.zScore, false, false } );
			    			    
				boxPlotPnl.setData(sd.names, getNormedSamples() );
				boxPlotPnl.plot();
				
			}
		} else if( ae.getSource() == addCCoords ) {
			
			List<double[]> ns = new ArrayList<double[]>();
			for( int i = 0; i < sd.samples.size(); i++ ) {
			    double[] a = sd.samples.get(i); 
				double[] b = Arrays.copyOf(a, a.length+2);
				b[a.length] = sd.geoms.get(i).getCentroid().getX();
				b[a.length+1] = sd.geoms.get(i).getCentroid().getY();
				ns.add(b);
			}
			sd.samples = ns;
			
			sd.names.add("x");
			sd.names.add("y");
			
			sd.bindings.add(SpatialDataFrame.binding.Double);
			sd.bindings.add(SpatialDataFrame.binding.Double);
			
			// get current norm-mode
			norm n = norm.none;
			for( int i = 0; i < dataTable.getRowCount(); i++ ) {
				if( (Boolean)dataTable.getValueAt(i, COORDINATE) ) {
					n = (norm) dataTable.getValueAt(i, NORM);
					break;
				}
			}
							
			dataTable.addRow( new Object[]{"x", n, true, true } );
			dataTable.addRow( new Object[]{"y", n, true, true } );
						
			addCCoords.setEnabled(false);
			
			boxPlotPnl.setData(sd.names, getNormedSamples() );
			boxPlotPnl.plot();
		} else if ( ae.getSource() == useAll ) {
			for( int i = 0; i < dataTable.getRowCount(); i++ )
				dataTable.setValueAt( !useAllState, i, USE );
			useAllState = !useAllState;
		} 
	}
	
	private boolean trainAllowed = false;
			
	@Override
	public void tableChanged(TableModelEvent e) {		
		if( e.getColumn() == COORDINATE || e.getColumn() == USE ) {
			boolean a = getFA().length > 0 && ( sd.geoms != null || getGA(true).length > 0 );
			firePropertyChange(TRAIN_ALLOWED_PROP, trainAllowed, a );
			trainAllowed = a;
		}
		
		dataTable.removeTableModelListener(this);
		// all used geoColumns must have the same normalization
		if( e.getColumn() == COORDINATE && (Boolean)dataTable.getValueAt(e.getFirstRow(), COORDINATE) ) {	
			// Set norm to the one of the others
			for( int i = 0; i < dataTable.getRowCount(); i++ ) {
				if( i == e.getFirstRow() )
					continue;
				if( (Boolean)dataTable.getValueAt(i, COORDINATE) ) {
					dataTable.setValueAt( dataTable.getValueAt(i, NORM), e.getFirstRow(), NORM );
					break;
				}
			}
		} else if( e.getColumn() == NORM && (Boolean)dataTable.getValueAt(e.getFirstRow(), COORDINATE) ) {
			// change norm of all other
			for( int i = 0; i < dataTable.getRowCount(); i++ ) {
				if( i == e.getFirstRow() ) //FIXME i is integer!!!
					continue;
				if( (Boolean)dataTable.getValueAt(i, COORDINATE) ) 
					dataTable.setValueAt( dataTable.getValueAt(e.getFirstRow(), NORM), i, NORM );
			}
		}
		dataTable.addTableModelListener(this);
		
		boxPlotPnl.setData(sd.names, getNormedSamples() );
		boxPlotPnl.plot();
	}
	
	public List<double[]> getNormedSamples() {
		if( sd == null ) { //TODO does this ever happen?
			log.error("No data loaded yet!");
			return null;
		}
		
		List<double[]> l = new ArrayList<double[]>();
		for( double[] d : sd.samples )
			l.add( Arrays.copyOf(d, d.length));
		
		boolean geoNormed = false;
		
		for( int i = 0; i < dataTable.getRowCount(); i++ ) {
			if( !(Boolean)dataTable.getValueAt(i, COORDINATE) ) {
				if( dataTable.getValueAt(i, NORM) == norm.scale )
					DataUtils.normalizeColumn(l, i);
				else if( dataTable.getValueAt(i, NORM) == norm.zScore )
					DataUtils.zScoreColumn(l, i);
			} else if( !geoNormed && (Boolean)dataTable.getValueAt(i, COORDINATE) ) { // only once
				int[] ga = getGA(true);
				if( dataTable.getValueAt(i, NORM) == norm.scale )
					DataUtils.normalizeGeoColumns(l, ga );
				else if( dataTable.getValueAt(i, NORM) == norm.zScore  )
					DataUtils.zScoreGeoColumns(l, ga, new EuclideanDist(ga) );
				geoNormed = true;	
			}
		}		
		return l;
	}
	
	public SpatialDataFrame getSpatialData() {
		if( sd.geoms == null ) { // no geoms present, if data stems from csv
			GeometryFactory gf = new GeometryFactory();
			sd.geoms = new ArrayList<Geometry>();
			
			int[] ga = getGA(true);
						
			for( double[] d : sd.samples ) {
				Coordinate c;
				if( ga.length == 2 )
					c = new Coordinate(d[ga[0]],d[ga[1]]);
				else if( ga.length == 3 )
					c = new Coordinate(d[ga[0]],d[ga[1]]);
				else 
					throw new RuntimeException("No spatial data submitted/found!");
				
				sd.geoms.add( gf.createPoint(c) );
			}
		}
		return sd;
	}
		
	public int[] getFA() {
		List<Integer> l = new ArrayList<Integer>();
		for( int i = 0; i < dataTable.getRowCount(); i++ ) {
			if( (Boolean)dataTable.getValueAt(i, USE) && !(Boolean)dataTable.getValueAt(i, COORDINATE) )
				l.add(i);
		}
		int[] fa = new int[l.size()];
		for( int i = 0; i < l.size(); i++ )
			fa[i] = l.get(i);
		return fa;
	}
	
	// Returns either the used GA or the complete GA, depending on flag
	public int[] getGA(boolean allGA) {
		List<Integer> l = new ArrayList<Integer>();
		for( int i = 0; i < dataTable.getRowCount(); i++ ) {
			if( ( allGA || (Boolean)dataTable.getValueAt(i, USE) ) && (Boolean)dataTable.getValueAt(i, COORDINATE) )
				l.add(i);
		}
		int[] ga = new int[l.size()];
		for( int i = 0; i < l.size(); i++ )
			ga[i] = l.get(i);
		return ga;
	}
}

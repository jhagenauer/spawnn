package spawnn.gui;

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
import javax.swing.filechooser.FileFilter;
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
	public static final String DATA_LOADED_PROP ="DATA_LOADED";
	public static final String COORD_CHANGED_PROP = "COORDINATES_CHANGED";
	static Set<String> coordNames;
	
	static {
		coordNames = new HashSet<String>();
		coordNames.add("x");
		coordNames.add("X");
		coordNames.add("y");
		coordNames.add("Y");
		coordNames.add("lon");
		coordNames.add("LON");
		coordNames.add("longitude");
		coordNames.add("lat");
		coordNames.add("LAT");
		coordNames.add("latitude");
	}
		
	private DefaultTableModel dataTable;
	private JButton btnLoadShp, addCCoords;
	private BoxPlotPanel boxPlotPnl;
	
	enum norm {none, scale, zScore};
	final int ATTRIBUTE = 0, NORM = 1, COORDINATE = 2, USE = 3;
	
	protected SpatialDataFrame sd = null;
	
	DataPanel() {
		super();
		
		btnLoadShp = new JButton("Load data...");
		btnLoadShp.addActionListener(this);
		setLayout(new MigLayout(""));
		add(btnLoadShp, "split 2");
		
		addCCoords = new JButton("Add centroid coordinates...");
		addCCoords.addActionListener(this);
		addCCoords.setEnabled(false);
		add(addCCoords, "wrap");
		
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
		
		//add( new JScrollPane(table), "push, grow" );	
		add( new JScrollPane(table), "w 50%, push, grow" );	
		
		boxPlotPnl = new BoxPlotPanel();
		add( boxPlotPnl, "w 50%, grow");
	}

	@Override
	public void actionPerformed(ActionEvent ae) {
		
		if( ae.getSource() == btnLoadShp ) {
			JFileChooser fc = new JFileChooser();
			
			FileFilter csvFilter = new FileFilter() {
				@Override
				public boolean accept(File f) {	return f.isDirectory() || f.getName().toLowerCase().endsWith(".csv"); }

				@Override
				public String getDescription() { return "Comma-separated values (*.csv)"; }
			};
			
			FileFilter shpFilter = new FileFilter() {
				@Override
				public boolean accept(File f) {	return f.isDirectory() || f.getName().toLowerCase().endsWith(".shp"); }

				@Override
				public String getDescription() { return "ESRI Shapefile (*.shp)"; }
			};
			fc.setFileFilter(csvFilter); // not supported by now. Problem: how to deal with coordinate vectors
			fc.setFileFilter(shpFilter);
									
			int state = fc.showOpenDialog(this);
			if( state == JFileChooser.APPROVE_OPTION ) {
				File fn = fc.getSelectedFile(); 
				
				if( fc.getFileFilter() == shpFilter ) {
			      sd = DataUtils.readSpatialDataFrameFromShapefile(fn, false);     
			      addCCoords.setEnabled(true);
				} else if( fc.getFileFilter() == csvFilter ) {				
					DataFrame df = DataUtils.readDataFrameFromCSV(fn, new int[]{}, true ); 
					sd = new SpatialDataFrame();				
					sd.samples = df.samples;
					sd.names = df.names;
					sd.bindings = df.bindings;
				}
				
				dataTable.setRowCount(0);
								
				int numCoords = 0;
			    for( String s : sd.names ) {
			    	if( coordNames.contains(s) ) {
			    		numCoords++;
			    		dataTable.addRow( new Object[]{s, norm.none, true, true } );
			    	} else {
			    		dataTable.addRow( new Object[]{s, norm.zScore, false, true } );
			    	}
			    }			    
				boxPlotPnl.setData(sd.names, getNormedSamples() );
				boxPlotPnl.plot();
				
				firePropertyChange(DATA_LOADED_PROP, false, true);
			    firePropertyChange(COORD_CHANGED_PROP, 0, numCoords);
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

			firePropertyChange(COORD_CHANGED_PROP, 0, 2);
		}
		
	}
	
	private int oldGALength = 0;
		
	@Override
	public void tableChanged(TableModelEvent e) {	
		if( e.getColumn() == COORDINATE ) {
			int newGALength = getGA().length;
			firePropertyChange(COORD_CHANGED_PROP, oldGALength, newGALength );
			oldGALength = newGALength;
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
				if( i == e.getFirstRow() )
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
		
		if( sd == null ) { //TODO is it ever called?
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
			} else if( !geoNormed && (Boolean)dataTable.getValueAt(i, COORDINATE) ) {
				int[] ga = getGA();
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
			
			int[] ga = getGA();
						
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
	
	// TODO does it make sense to mark them as coordinates and don't use them?
	public int[] getGA() {
		List<Integer> l = new ArrayList<Integer>();
		for( int i = 0; i < dataTable.getRowCount(); i++ ) {
			if( (Boolean)dataTable.getValueAt(i, USE) && (Boolean)dataTable.getValueAt(i, COORDINATE) )
				l.add(i);
		}
		int[] ga = new int[l.size()];
		for( int i = 0; i < l.size(); i++ )
			ga[i] = l.get(i);
		return ga;
	}
}

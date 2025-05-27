import numpy as np
from netCDF4 import Dataset as NCFile

class CamraLoader:
    """Loads a raw CAMRa .nc file"""
    def __init__(self, file):
        nc = NCFile(file)
        print(nc.ncattrs())
        print(nc.variables.keys())

        # Attributes
        self.instrument_name = nc.getncattr('instrument_name')
        self.field_names = nc.getncattr('field_names')
        self.scan_name = nc.getncattr('scan_name')
        self.last_revised_date = nc.getncattr('last_revised_date')
        self.conventions = nc.getncattr('Conventions')
        self.title = nc.getncattr('title')
        self.institution = nc.getncattr('institution')
        self.references = nc.getncattr('references')
        self.source = nc.getncattr('source')
        self.product_version = nc.getncattr('product_version')
        self.processing_level = nc.getncattr('processing_level')
        self.licence = nc.getncattr('licence')
        self.acknowledgement = nc.getncattr('acknowledgement')
        self.platform = nc.getncattr('platform')
        self.platform_type = nc.getncattr('platform_type')
        self.location_keywords = nc.getncattr('location_keywords')
        self.platform_is_mobile = nc.getncattr('platform_is_mobile')
        self.deployment_mode = nc.getncattr('deployment_mode')
        self.platform_altitude = nc.getncattr('platform_altitude')
        self.creator_name = nc.getncattr('creator_name')
        self.creator_email = nc.getncattr('creator_email')
        self.creator_url = nc.getncattr('creator_url')
        self.instrument_software = nc.getncattr('instrument_software')
        self.instrument_software_version = nc.getncattr('instrument_software_version')
        self.instrument_manufacturer = nc.getncattr('instrument_manufacturer')
        self.instrument_model = nc.getncattr('instrument_model')
        self.instrument_serial_number = nc.getncattr('instrument_serial_number')
        self.instrument_pid = nc.getncattr('instrument_pid')
        self.project = nc.getncattr('project')
        self.project_principal_investigator = nc.getncattr('project_principal_investigator')
        self.project_principal_investigator_email = nc.getncattr('project_principal_investigator_email')
        self.project_principal_investigator_url = nc.getncattr('project_principal_investigator_url')
        self.processing_software_url = nc.getncattr('processing_software_url')
        self.processing_software_version = nc.getncattr('processing_software_version')
        self.time_coverage_start = nc.getncattr('time_coverage_start')
        self.time_coverage_end = nc.getncattr('time_coverage_end')
        self.geospatial_bounds = nc.getncattr('geospatial_bounds')
        self.history = nc.getncattr('history')
        self.comment = nc.getncattr('comment')

        # Data fields
        self.time = nc.variables['time'][:]
        self.range = nc.variables['range'][:]
        self.azimuth = nc.variables['azimuth'][:]
        self.elevation = nc.variables['elevation'][:]
        self.dbzh = nc.variables['DBZ_H'][:]
        self.vel_h = nc.variables['VEL_H'][:]
        self.vel_v = nc.variables['VEL_V'][:]
        self.vel_hv = nc.variables['VEL_HV'][:]
        self.zdr = nc.variables['ZDR'][:]
        self.ldr = nc.variables['LDR'][:]
        self.spw_hv = nc.variables['SPW_HV'][:]
        self.phidp = nc.variables['PHIDP'][:]
        self.cxc = nc.variables['CXC'][:]
        self.ddv = nc.variables['DDV'][:]
        self.phi_hv = nc.variables['PHI_HV'][:]
        self.phi_hvd = nc.variables['PHI_HVD'][:]
        self.snr_h = nc.variables['SNR_H'][:]
        self.snr_v = nc.variables['SNR_V'][:]
        self.snr_x = nc.variables['SNR_X'][:]
        self.sweep_number = nc.variables['sweep_number'][:]
        self.fixed_angle = nc.variables['fixed_angle'][:]
        self.sweep_start_ray_index = nc.variables['sweep_start_ray_index'][:]
        self.sweep_end_ray_index = nc.variables['sweep_end_ray_index'][:]
        self.sweep_mode = nc.variables['sweep_mode'][:]
        self.prt = nc.variables['prt'][:]
        self.pulse_width = nc.variables['pulse_width'][:]
        self.radar_beam_width_h = nc.variables['radar_beam_width_h'][:]
        self.radar_beam_width_v = nc.variables['radar_beam_width_v'][:]
        self.latitude = nc.variables['latitude'][:]
        self.longitude = nc.variables['longitude'][:]
        self.altitude = nc.variables['altitude'][:]
        self.time_coverage_start = nc.variables['time_coverage_start'][:]
        self.time_coverage_end = nc.variables['time_coverage_end'][:]
        self.time_reference = nc.variables['time_reference'][:]
        self.volume_number = nc.variables['volume_number'][:]
        self.altitude_agl = nc.variables['altitude_agl'][:]

        self._calculate_radar_coords()

    def _calculate_radar_coords(self):
        r_earth = 6371.0 * (4.0 / 3.0) * 1000
        range_mat = np.mat(self.range)
        azim_rad_mat = np.deg2rad(np.mat(self.azimuth))
        elev_rad_mat = np.deg2rad(np.mat(self.elevation))

        self.r_coords = np.asarray(np.dot(np.cos(np.transpose(elev_rad_mat)), range_mat))
        self.x_coords = np.asarray(np.dot(np.transpose(np.multiply(np.sin(azim_rad_mat), np.cos(elev_rad_mat))), range_mat))
        self.y_coords = np.asarray(np.dot(np.transpose(np.multiply(np.cos(azim_rad_mat), np.cos(elev_rad_mat))), range_mat))
        self.z_coords = np.asarray(np.dot(np.transpose(np.sin(elev_rad_mat)), range_mat) + np.sqrt(self.r_coords ** 2 + r_earth ** 2) - r_earth)

if __name__ == '__main__':
    cl = CamraLoader(file="../../data/ncas-radar-camra-1_cao_20230612-130655_rhi_l1_v1.0.0.nc")
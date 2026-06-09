"""Canonical identifiers and stellar parameters for the project's targets.

Values here should be treated as the single source of truth used by loaders,
plotting, and modelling code. Update the source citation in PROJECT.md when
a value changes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Star:
    name: str
    hd: str
    hip: str
    gj: str | None
    gaia_dr3_source_id: int
    ra_deg: float
    dec_deg: float
    distance_pc: float
    spectral_type: str
    v_mag: float
    mass_msun: float
    radius_rsun: float
    rotation_period_days: float | None
    notes: str = ""


TAU_CETI = Star(
    name="Tau Ceti",
    hd="HD 10700",
    hip="HIP 8102",
    gj="GJ 71",
    # NOTE: verify with SIMBAD before use in production analysis.
    gaia_dr3_source_id=2452378776434276992,
    ra_deg=26.017,
    dec_deg=-15.937,
    distance_pc=3.65,
    spectral_type="G8V",
    v_mag=3.50,
    mass_msun=0.78,
    radius_rsun=0.79,
    rotation_period_days=46.0,
    notes="Old metal-poor G dwarf; resolved inner debris disk (ALMA, MacGregor+2016).",
)


EPS_ERI = Star(
    name="epsilon Eridani",
    hd="HD 22049",
    hip="HIP 16537",
    gj="GJ 144",
    # NOTE: verify with SIMBAD before use in production analysis.
    gaia_dr3_source_id=5164707970261890560,
    ra_deg=53.232,
    dec_deg=-9.458,
    distance_pc=3.21,
    spectral_type="K2V",
    v_mag=3.73,
    mass_msun=0.82,
    radius_rsun=0.74,
    rotation_period_days=11.45,
    notes="Young (~0.8 Gyr) active K dwarf; direct-imaged planet eps Eri b at "
          "~7yr orbit (Mawet+2019, Llop-Sayson+2021). Resolved debris disk.",
)

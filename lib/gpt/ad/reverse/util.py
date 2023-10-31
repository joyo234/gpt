#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import gpt as g


def is_field(x):
    if isinstance(x, g.lattice):
        return True
    elif isinstance(x, g.tensor):
        return False
    elif isinstance(x, g.expr):
        return x.lattice() is not None
    else:
        raise Exception(f"Unknown object type {type(x)}")


def accumulate(lhs, rhs):
    if is_field(rhs) and not is_field(lhs):
        rhs = g.sum(rhs)
    lhs += rhs